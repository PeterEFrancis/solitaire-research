#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int kSuits = 2;
constexpr int kMaxCards = 26;
constexpr int kMaxTableau = 7;
constexpr uint8_t kFaceUp = 0x80;
constexpr uint8_t kCardMask = 0x7f;
constexpr size_t kCacheLimit = 1'000'000;

struct Pile {
    std::array<uint8_t, kMaxCards> cards{};
    uint8_t size = 0;
};

struct State {
    std::array<Pile, kMaxTableau> tableau{};
    std::array<uint8_t, kMaxCards> reserve{};
    std::array<uint8_t, kSuits> foundations{};
    uint8_t reserve_size = 0;
};

struct OrderedState {
    std::array<Pile, kMaxTableau> tableau{};
    std::array<uint8_t, kMaxCards> stock{};
    std::array<uint8_t, kMaxCards> waste{};
    std::array<uint8_t, kSuits> foundations{};
    uint8_t stock_size = 0;
    uint8_t waste_size = 0;
    uint8_t recycles = 0;
};

struct Key {
    std::array<uint8_t, 72> bytes{};

    bool operator==(const Key& other) const { return bytes == other.bytes; }
    bool operator<(const Key& other) const { return bytes < other.bytes; }
};

struct KeyHash {
    size_t operator()(const Key& key) const {
        uint64_t hash = 1469598103934665603ULL;
        for (uint8_t byte : key.bytes) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
        return static_cast<size_t>(hash);
    }
};

struct Candidate {
    int score = 0;
    State state{};
    int removed_reserve_card = -1;
};

struct SearchNode {
    State state{};
    Key key{};
    uint32_t parent = UINT32_MAX;
};

struct OrderedCandidate {
    int score = 0;
    OrderedState state{};
};

struct OrderedSearchNode {
    OrderedState state{};
    Key key{};
    uint32_t parent = UINT32_MAX;
};

struct RemovalOrder {
    std::array<uint8_t, kMaxCards> cards{};
    uint8_t size = 0;

    void push(uint8_t card) {
        cards[size++] = card & kCardMask;
    }

    std::vector<uint8_t> to_vector() const {
        return std::vector<uint8_t>(
            cards.begin(),
            cards.begin() + size
        );
    }
};

int card_id(uint8_t card) { return card & kCardMask; }
bool face_up(uint8_t card) { return (card & kFaceUp) != 0; }
uint8_t make_face_up(uint8_t card) { return card | kFaceUp; }

uint64_t factorial(int value) {
    uint64_t result = 1;
    for (int factor = 2; factor <= value; ++factor) {
        result *= static_cast<uint64_t>(factor);
    }
    return result;
}

int default_tableau_count(int n) {
    const int deck_size = kSuits * n;
    int best_columns = 1;
    int best_difference = std::abs(2 - deck_size);
    for (int columns = 2;
         columns * (columns + 1) / 2 <= deck_size;
         ++columns) {
        int difference = std::abs(
            columns * (columns + 1) - deck_size
        );
        // Prefer the larger tableau when both splits are equally balanced.
        if (difference <= best_difference) {
            best_columns = columns;
            best_difference = difference;
        }
    }
    return best_columns;
}

int stock_card_count(int n) {
    int columns = default_tableau_count(n);
    return kSuits * n - columns * (columns + 1) / 2;
}

bool pile_equal(const Pile& left, const Pile& right) {
    return left.size == right.size &&
        std::equal(
            left.cards.begin(),
            left.cards.begin() + left.size,
            right.cards.begin()
        );
}

class Solver {
public:
    Solver(int n, int tableau_count)
        : n_(n), tableau_count_(tableau_count) {}

    bool is_solvable(State initial) {
        normalize_safe_foundations(initial);
        Key initial_key = position_key(initial);
        if (is_won(initial) || known_solvable_.find(initial_key) != known_solvable_.end()) {
            ++solvable_cache_hits;
            return true;
        }
        if (known_unsolvable_.find(initial_key) != known_unsolvable_.end()) {
            ++unsolvable_cache_hits;
            return false;
        }

        std::vector<SearchNode> nodes;
        nodes.reserve(256);
        std::vector<uint32_t> pending;
        pending.reserve(256);
        std::unordered_map<Key, uint32_t, KeyHash> visited;
        visited.reserve(256);

        nodes.push_back({std::move(initial), initial_key, UINT32_MAX});
        pending.push_back(0);
        visited.emplace(initial_key, 0);

        std::vector<Candidate> successors;
        successors.reserve(48);

        while (!pending.empty()) {
            uint32_t node_index = pending.back();
            pending.pop_back();
            const SearchNode& node = nodes[node_index];

            if (known_solvable_.find(node.key) != known_solvable_.end()) {
                mark_path_solvable(nodes, node_index);
                ++solvable_cache_hits;
                return true;
            }
            if (is_won(node.state)) {
                mark_path_solvable(nodes, node_index);
                return true;
            }

            ++expanded_positions;
            successors.clear();
            generate_successors(node.state, successors);
            std::sort(
                successors.begin(),
                successors.end(),
                [](const Candidate& left, const Candidate& right) {
                    return left.score < right.score;
                }
            );

            for (Candidate& candidate : successors) {
                normalize_safe_foundations(candidate.state);
                Key key = position_key(candidate.state);
                if (known_solvable_.find(key) != known_solvable_.end() ||
                    is_won(candidate.state)) {
                    mark_path_solvable(nodes, node_index);
                    known_solvable_.insert(key);
                    return true;
                }
                if (known_unsolvable_.find(key) != known_unsolvable_.end() ||
                    visited.find(key) != visited.end()) {
                    continue;
                }
                uint32_t child_index = static_cast<uint32_t>(nodes.size());
                visited.emplace(key, child_index);
                nodes.push_back(
                    {std::move(candidate.state), key, node_index}
                );
                pending.push_back(child_index);
            }
        }

        if (known_unsolvable_.size() + nodes.size() > kCacheLimit) {
            known_unsolvable_.clear();
        }
        for (const SearchNode& node : nodes) {
            known_unsolvable_.insert(node.key);
        }
        return false;
    }

    bool find_solution(
        State initial,
        std::vector<uint8_t>& reserve_order,
        int variation = 0
    ) {
        struct CertificateNode {
            State state{};
            Key key{};
            RemovalOrder order{};
        };

        RemovalOrder initial_order;
        normalize_safe_foundations(
            initial,
            &initial_order,
            variation % 2 == 1
        );
        Key initial_key = position_key(initial);
        if (is_won(initial)) {
            reserve_order = initial_order.to_vector();
            return true;
        }

        std::vector<CertificateNode> nodes;
        nodes.reserve(256);
        std::vector<uint32_t> pending;
        pending.reserve(256);
        std::unordered_set<Key, KeyHash> visited;
        visited.reserve(256);
        nodes.push_back({std::move(initial), initial_key, initial_order});
        pending.push_back(0);
        visited.insert(initial_key);

        std::vector<Candidate> successors;
        successors.reserve(48);
        while (!pending.empty()) {
            uint32_t node_index = pending.back();
            pending.pop_back();
            const CertificateNode& node = nodes[node_index];
            RemovalOrder parent_order = node.order;

            ++expanded_positions;
            successors.clear();
            generate_successors(node.state, successors);
            std::sort(
                successors.begin(),
                successors.end(),
                [variation](
                    const Candidate& left,
                    const Candidate& right
                ) {
                    if (left.score != right.score) {
                        return variation < 2
                            ? left.score < right.score
                            : left.score > right.score;
                    }
                    return variation % 2 == 0
                        ? left.removed_reserve_card <
                            right.removed_reserve_card
                        : left.removed_reserve_card >
                            right.removed_reserve_card;
                }
            );

            for (Candidate& candidate : successors) {
                RemovalOrder order = parent_order;
                if (candidate.removed_reserve_card >= 0) {
                    order.push(static_cast<uint8_t>(
                        candidate.removed_reserve_card
                    ));
                }
                normalize_safe_foundations(
                    candidate.state,
                    &order,
                    variation % 2 == 1
                );
                Key key = position_key(candidate.state);
                if (is_won(candidate.state)) {
                    reserve_order = order.to_vector();
                    return true;
                }
                if (!visited.insert(key).second) {
                    continue;
                }
                nodes.push_back({
                    std::move(candidate.state),
                    key,
                    order,
                });
                pending.push_back(
                    static_cast<uint32_t>(nodes.size() - 1)
                );
            }
        }
        return false;
    }

    uint64_t expanded_positions = 0;
    uint64_t normalized_foundation_moves = 0;
    uint64_t solvable_cache_hits = 0;
    uint64_t unsolvable_cache_hits = 0;

private:
    int n_;
    int tableau_count_;
    std::unordered_set<Key, KeyHash> known_solvable_;
    std::unordered_set<Key, KeyHash> known_unsolvable_;

    int suit(uint8_t card) const { return card_id(card) / n_; }
    int rank(uint8_t card) const { return card_id(card) % n_ + 1; }

    bool is_won(const State& state) const {
        return state.foundations[0] == n_ && state.foundations[1] == n_;
    }

    bool can_foundation(const State& state, uint8_t card) const {
        return face_up(card) &&
            state.foundations[suit(card)] + 1 == rank(card);
    }

    bool safe_foundation(const State& state, uint8_t card) const {
        if (rank(card) == 1) {
            return true;
        }
        int other_suit = 1 - suit(card);
        return state.foundations[other_suit] >= rank(card) - 1;
    }

    bool can_build(uint8_t card, const Pile& destination) const {
        if (!face_up(card)) {
            return false;
        }
        if (destination.size == 0) {
            return rank(card) == n_;
        }
        uint8_t top = destination.cards[destination.size - 1];
        return face_up(top) && rank(card) + 1 == rank(top);
    }

    bool packed_stack(const Pile& pile, int start) const {
        for (int index = start; index < pile.size; ++index) {
            if (!face_up(pile.cards[index])) {
                return false;
            }
            if (index + 1 < pile.size &&
                rank(pile.cards[index]) != rank(pile.cards[index + 1]) + 1) {
                return false;
            }
        }
        return true;
    }

    void reveal_top(Pile& pile) const {
        if (pile.size > 0) {
            pile.cards[pile.size - 1] =
                make_face_up(pile.cards[pile.size - 1]);
        }
    }

    void remove_reserve_card(State& state, int index) const {
        for (int position = index; position + 1 < state.reserve_size; ++position) {
            state.reserve[position] = state.reserve[position + 1];
        }
        --state.reserve_size;
    }

    void normalize_safe_foundations(
        State& state,
        RemovalOrder* reserve_order = nullptr,
        bool prefer_high_card = false
    ) {
        while (true) {
            int reserve_index = -1;
            uint8_t best_card = UINT8_MAX;
            for (int index = 0; index < state.reserve_size; ++index) {
                uint8_t card = make_face_up(state.reserve[index]);
                if (can_foundation(state, card) &&
                    safe_foundation(state, card) &&
                    (best_card == UINT8_MAX ||
                     (prefer_high_card
                        ? card_id(card) > card_id(best_card)
                        : card_id(card) < card_id(best_card)))) {
                    reserve_index = index;
                    best_card = card;
                }
            }
            if (reserve_index >= 0) {
                if (reserve_order != nullptr) {
                    reserve_order->push(best_card);
                }
                remove_reserve_card(state, reserve_index);
                ++state.foundations[suit(best_card)];
                ++normalized_foundation_moves;
                continue;
            }

            int source = -1;
            best_card = UINT8_MAX;
            for (int column = 0; column < tableau_count_; ++column) {
                const Pile& pile = state.tableau[column];
                if (pile.size == 0) {
                    continue;
                }
                uint8_t card = pile.cards[pile.size - 1];
                if (can_foundation(state, card) &&
                    safe_foundation(state, card) &&
                    (best_card == UINT8_MAX ||
                     (prefer_high_card
                        ? card_id(card) > card_id(best_card)
                        : card_id(card) < card_id(best_card)))) {
                    source = column;
                    best_card = card;
                }
            }
            if (source < 0) {
                return;
            }
            --state.tableau[source].size;
            reveal_top(state.tableau[source]);
            ++state.foundations[suit(best_card)];
            ++normalized_foundation_moves;
        }
    }

    uint8_t transform_card(uint8_t card, bool swap_suits) const {
        int id = card_id(card);
        int card_suit = id / n_;
        int rank_index = id % n_;
        int mapped_suit = swap_suits ? 1 - card_suit : card_suit;
        uint8_t mapped = static_cast<uint8_t>(mapped_suit * n_ + rank_index);
        return face_up(card) ? make_face_up(mapped) : mapped;
    }

    Key position_key_for_mapping(const State& state, bool swap_suits) const {
        std::array<Pile, kMaxTableau> piles{};
        for (int column = 0; column < tableau_count_; ++column) {
            piles[column].size = state.tableau[column].size;
            for (int index = 0; index < piles[column].size; ++index) {
                piles[column].cards[index] =
                    transform_card(state.tableau[column].cards[index], swap_suits);
            }
        }
        std::sort(
            piles.begin(),
            piles.begin() + tableau_count_,
            [](const Pile& left, const Pile& right) {
                int common = std::min(left.size, right.size);
                for (int index = 0; index < common; ++index) {
                    if (left.cards[index] != right.cards[index]) {
                        return left.cards[index] < right.cards[index];
                    }
                }
                return left.size < right.size;
            }
        );

        Key key{};
        size_t cursor = 0;
        for (int column = 0; column < tableau_count_; ++column) {
            const Pile& pile = piles[column];
            key.bytes[cursor++] = pile.size;
            for (int index = 0; index < pile.size; ++index) {
                key.bytes[cursor++] = pile.cards[index];
            }
        }

        key.bytes[cursor++] = state.reserve_size;
        std::array<uint8_t, kMaxCards> reserve{};
        for (int index = 0; index < state.reserve_size; ++index) {
            reserve[index] = transform_card(state.reserve[index], swap_suits);
        }
        std::sort(reserve.begin(), reserve.begin() + state.reserve_size);
        for (int index = 0; index < state.reserve_size; ++index) {
            key.bytes[cursor++] = reserve[index];
        }
        key.bytes[cursor++] =
            state.foundations[swap_suits ? 1 : 0];
        key.bytes[cursor++] =
            state.foundations[swap_suits ? 0 : 1];
        return key;
    }

    Key position_key(const State& state) const {
        Key identity = position_key_for_mapping(state, false);
        Key swapped = position_key_for_mapping(state, true);
        return std::min(identity, swapped);
    }

    void mark_path_solvable(
        const std::vector<SearchNode>& nodes,
        uint32_t node_index
    ) {
        if (known_solvable_.size() + nodes.size() > kCacheLimit) {
            known_solvable_.clear();
        }
        uint32_t current = node_index;
        while (current != UINT32_MAX) {
            known_solvable_.insert(nodes[current].key);
            current = nodes[current].parent;
        }
    }

    bool destination_already_used(
        const State& state,
        int destination,
        const std::vector<int>& used
    ) const {
        return std::any_of(
            used.begin(),
            used.end(),
            [&state, destination](int previous) {
                return pile_equal(
                    state.tableau[destination],
                    state.tableau[previous]
                );
            }
        );
    }

    void generate_successors(
        const State& state,
        std::vector<Candidate>& result
    ) const {
        for (int reserve_index = 0;
             reserve_index < state.reserve_size;
             ++reserve_index) {
            uint8_t card = make_face_up(state.reserve[reserve_index]);
            if (can_foundation(state, card)) {
                State next = state;
                remove_reserve_card(next, reserve_index);
                ++next.foundations[suit(card)];
                result.push_back({
                    500,
                    std::move(next),
                    card_id(card),
                });
            }

            std::vector<int> used_destinations;
            used_destinations.reserve(tableau_count_);
            for (int destination = 0;
                 destination < tableau_count_;
                 ++destination) {
                if (!can_build(card, state.tableau[destination]) ||
                    destination_already_used(
                        state,
                        destination,
                        used_destinations
                    )) {
                    continue;
                }
                used_destinations.push_back(destination);
                State next = state;
                remove_reserve_card(next, reserve_index);
                next.tableau[destination]
                    .cards[next.tableau[destination].size++] = card;
                int score = state.tableau[destination].size == 0 ? 180 : 120;
                result.push_back({
                    score,
                    std::move(next),
                    card_id(card),
                });
            }
        }

        for (int source = 0; source < tableau_count_; ++source) {
            const Pile& source_pile = state.tableau[source];
            if (source_pile.size == 0 ||
                !face_up(source_pile.cards[source_pile.size - 1])) {
                continue;
            }

            uint8_t top = source_pile.cards[source_pile.size - 1];
            if (can_foundation(state, top)) {
                State next = state;
                --next.tableau[source].size;
                bool reveals = next.tableau[source].size > 0 &&
                    !face_up(
                        next.tableau[source]
                            .cards[next.tableau[source].size - 1]
                    );
                reveal_top(next.tableau[source]);
                ++next.foundations[suit(top)];
                result.push_back({reveals ? 1400 : 550, std::move(next)});
            }

            for (int start = 0; start < source_pile.size; ++start) {
                if (!packed_stack(source_pile, start)) {
                    continue;
                }
                uint8_t moving_card = source_pile.cards[start];
                std::vector<int> used_destinations;
                used_destinations.reserve(tableau_count_);
                for (int destination = 0;
                     destination < tableau_count_;
                     ++destination) {
                    if (destination == source ||
                        !can_build(
                            moving_card,
                            state.tableau[destination]
                        ) ||
                        destination_already_used(
                            state,
                            destination,
                            used_destinations
                        )) {
                        continue;
                    }
                    if (state.tableau[destination].size == 0 && start == 0) {
                        // Moving a whole pile to an empty pile only renames piles.
                        continue;
                    }
                    used_destinations.push_back(destination);

                    State next = state;
                    Pile& next_source = next.tableau[source];
                    Pile& next_destination = next.tableau[destination];
                    for (int index = start; index < source_pile.size; ++index) {
                        next_destination.cards[next_destination.size++] =
                            source_pile.cards[index];
                    }
                    next_source.size = static_cast<uint8_t>(start);
                    bool reveals = next_source.size > 0 &&
                        !face_up(next_source.cards[next_source.size - 1]);
                    bool empties = next_source.size == 0;
                    reveal_top(next_source);
                    int moved = source_pile.size - start;
                    int score = reveals
                        ? 1300 + moved
                        : (empties ? 700 + moved : 40 + moved);
                    result.push_back({score, std::move(next)});
                }
            }
        }
    }
};

class OrderedSolver {
public:
    OrderedSolver(int n, int tableau_count, int max_recycles = 3)
        : n_(n),
          tableau_count_(tableau_count),
          max_recycles_(max_recycles) {}

    bool is_solvable(OrderedState initial) {
        normalize_safe_foundations(initial);
        Key initial_key = position_key(initial);
        if (is_won(initial) ||
            known_solvable_.find(initial_key) != known_solvable_.end()) {
            ++solvable_cache_hits;
            return true;
        }
        if (known_unsolvable_.find(initial_key) != known_unsolvable_.end()) {
            ++unsolvable_cache_hits;
            return false;
        }

        std::vector<OrderedSearchNode> nodes;
        nodes.reserve(256);
        std::vector<uint32_t> pending;
        pending.reserve(256);
        std::unordered_map<Key, uint32_t, KeyHash> visited;
        visited.reserve(256);
        nodes.push_back({std::move(initial), initial_key, UINT32_MAX});
        pending.push_back(0);
        visited.emplace(initial_key, 0);

        std::vector<OrderedCandidate> successors;
        successors.reserve(40);
        while (!pending.empty()) {
            uint32_t node_index = pending.back();
            pending.pop_back();
            const OrderedSearchNode& node = nodes[node_index];
            if (known_solvable_.find(node.key) != known_solvable_.end()) {
                mark_path_solvable(nodes, node_index);
                ++solvable_cache_hits;
                return true;
            }
            if (is_won(node.state)) {
                mark_path_solvable(nodes, node_index);
                return true;
            }

            ++expanded_positions;
            successors.clear();
            generate_successors(node.state, successors);
            std::sort(
                successors.begin(),
                successors.end(),
                [](const OrderedCandidate& left,
                   const OrderedCandidate& right) {
                    return left.score < right.score;
                }
            );
            for (OrderedCandidate& candidate : successors) {
                normalize_safe_foundations(candidate.state);
                Key key = position_key(candidate.state);
                if (known_solvable_.find(key) != known_solvable_.end() ||
                    is_won(candidate.state)) {
                    mark_path_solvable(nodes, node_index);
                    known_solvable_.insert(key);
                    return true;
                }
                if (known_unsolvable_.find(key) != known_unsolvable_.end() ||
                    visited.find(key) != visited.end()) {
                    continue;
                }
                uint32_t child_index =
                    static_cast<uint32_t>(nodes.size());
                visited.emplace(key, child_index);
                nodes.push_back({
                    std::move(candidate.state),
                    key,
                    node_index,
                });
                pending.push_back(child_index);
            }
        }

        if (known_unsolvable_.size() + nodes.size() > kCacheLimit) {
            known_unsolvable_.clear();
        }
        for (const OrderedSearchNode& node : nodes) {
            known_unsolvable_.insert(node.key);
        }
        return false;
    }

    uint64_t expanded_positions = 0;
    uint64_t normalized_foundation_moves = 0;
    uint64_t solvable_cache_hits = 0;
    uint64_t unsolvable_cache_hits = 0;

private:
    int n_;
    int tableau_count_;
    int max_recycles_;
    std::unordered_set<Key, KeyHash> known_solvable_;
    std::unordered_set<Key, KeyHash> known_unsolvable_;

    int suit(uint8_t card) const { return card_id(card) / n_; }
    int rank(uint8_t card) const { return card_id(card) % n_ + 1; }

    bool is_won(const OrderedState& state) const {
        return state.foundations[0] == n_ &&
            state.foundations[1] == n_;
    }

    bool can_foundation(
        const OrderedState& state,
        uint8_t card
    ) const {
        return face_up(card) &&
            state.foundations[suit(card)] + 1 == rank(card);
    }

    bool safe_foundation(
        const OrderedState& state,
        uint8_t card
    ) const {
        if (rank(card) == 1) {
            return true;
        }
        return state.foundations[1 - suit(card)] >= rank(card) - 1;
    }

    bool can_build(uint8_t card, const Pile& destination) const {
        if (!face_up(card)) {
            return false;
        }
        if (destination.size == 0) {
            return rank(card) == n_;
        }
        uint8_t top = destination.cards[destination.size - 1];
        return face_up(top) && rank(card) + 1 == rank(top);
    }

    bool packed_stack(const Pile& pile, int start) const {
        for (int index = start; index < pile.size; ++index) {
            if (!face_up(pile.cards[index])) {
                return false;
            }
            if (index + 1 < pile.size &&
                rank(pile.cards[index]) !=
                    rank(pile.cards[index + 1]) + 1) {
                return false;
            }
        }
        return true;
    }

    void reveal_top(Pile& pile) const {
        if (pile.size > 0) {
            pile.cards[pile.size - 1] =
                make_face_up(pile.cards[pile.size - 1]);
        }
    }

    void normalize_safe_foundations(OrderedState& state) {
        while (true) {
            int source = -1;
            uint8_t best_card = UINT8_MAX;
            if (state.waste_size > 0) {
                uint8_t card = make_face_up(
                    state.waste[state.waste_size - 1]
                );
                if (can_foundation(state, card) &&
                    safe_foundation(state, card)) {
                    source = -2;
                    best_card = card;
                }
            }
            for (int column = 0; column < tableau_count_; ++column) {
                const Pile& pile = state.tableau[column];
                if (pile.size == 0) {
                    continue;
                }
                uint8_t card = pile.cards[pile.size - 1];
                if (can_foundation(state, card) &&
                    safe_foundation(state, card) &&
                    (best_card == UINT8_MAX ||
                     card_id(card) < card_id(best_card))) {
                    source = column;
                    best_card = card;
                }
            }
            if (source == -1) {
                return;
            }
            if (source == -2) {
                --state.waste_size;
            } else {
                --state.tableau[source].size;
                reveal_top(state.tableau[source]);
            }
            ++state.foundations[suit(best_card)];
            ++normalized_foundation_moves;
        }
    }

    uint8_t transform_card(uint8_t card, bool swap_suits) const {
        int id = card_id(card);
        int card_suit = id / n_;
        int rank_index = id % n_;
        int mapped_suit = swap_suits ? 1 - card_suit : card_suit;
        uint8_t mapped = static_cast<uint8_t>(
            mapped_suit * n_ + rank_index
        );
        return face_up(card) ? make_face_up(mapped) : mapped;
    }

    Key position_key_for_mapping(
        const OrderedState& state,
        bool swap_suits
    ) const {
        std::array<Pile, kMaxTableau> piles{};
        for (int column = 0; column < tableau_count_; ++column) {
            piles[column].size = state.tableau[column].size;
            for (int index = 0; index < piles[column].size; ++index) {
                piles[column].cards[index] = transform_card(
                    state.tableau[column].cards[index],
                    swap_suits
                );
            }
        }
        std::sort(
            piles.begin(),
            piles.begin() + tableau_count_,
            [](const Pile& left, const Pile& right) {
                int common = std::min(left.size, right.size);
                for (int index = 0; index < common; ++index) {
                    if (left.cards[index] != right.cards[index]) {
                        return left.cards[index] < right.cards[index];
                    }
                }
                return left.size < right.size;
            }
        );

        Key key{};
        size_t cursor = 0;
        for (int column = 0; column < tableau_count_; ++column) {
            const Pile& pile = piles[column];
            key.bytes[cursor++] = pile.size;
            for (int index = 0; index < pile.size; ++index) {
                key.bytes[cursor++] = pile.cards[index];
            }
        }
        key.bytes[cursor++] = state.stock_size;
        for (int index = 0; index < state.stock_size; ++index) {
            key.bytes[cursor++] = transform_card(
                state.stock[index],
                swap_suits
            );
        }
        key.bytes[cursor++] = state.waste_size;
        for (int index = 0; index < state.waste_size; ++index) {
            key.bytes[cursor++] = transform_card(
                state.waste[index],
                swap_suits
            );
        }
        key.bytes[cursor++] =
            state.foundations[swap_suits ? 1 : 0];
        key.bytes[cursor++] =
            state.foundations[swap_suits ? 0 : 1];
        key.bytes[cursor++] = state.recycles;
        if (cursor > key.bytes.size()) {
            throw std::runtime_error("ordered state key overflow");
        }
        return key;
    }

    Key position_key(const OrderedState& state) const {
        Key identity = position_key_for_mapping(state, false);
        Key swapped = position_key_for_mapping(state, true);
        return std::min(identity, swapped);
    }

    void mark_path_solvable(
        const std::vector<OrderedSearchNode>& nodes,
        uint32_t node_index
    ) {
        if (known_solvable_.size() + nodes.size() > kCacheLimit) {
            known_solvable_.clear();
        }
        uint32_t current = node_index;
        while (current != UINT32_MAX) {
            known_solvable_.insert(nodes[current].key);
            current = nodes[current].parent;
        }
    }

    bool destination_already_used(
        const OrderedState& state,
        int destination,
        const std::vector<int>& used
    ) const {
        return std::any_of(
            used.begin(),
            used.end(),
            [&state, destination](int previous) {
                return pile_equal(
                    state.tableau[destination],
                    state.tableau[previous]
                );
            }
        );
    }

    void generate_successors(
        const OrderedState& state,
        std::vector<OrderedCandidate>& result
    ) const {
        if (state.stock_size > 0) {
            OrderedState next = state;
            uint8_t card = next.stock[next.stock_size - 1];
            --next.stock_size;
            next.waste[next.waste_size++] = card;
            result.push_back({20, std::move(next)});
        } else if (state.waste_size > 0 &&
                   state.recycles < max_recycles_) {
            OrderedState next = state;
            for (int index = 0; index < state.waste_size; ++index) {
                next.stock[index] =
                    state.waste[state.waste_size - 1 - index];
            }
            next.stock_size = state.waste_size;
            next.waste_size = 0;
            ++next.recycles;
            result.push_back({10, std::move(next)});
        }

        if (state.waste_size > 0) {
            uint8_t card = make_face_up(
                state.waste[state.waste_size - 1]
            );
            if (can_foundation(state, card)) {
                OrderedState next = state;
                --next.waste_size;
                ++next.foundations[suit(card)];
                result.push_back({500, std::move(next)});
            }

            std::vector<int> used_destinations;
            used_destinations.reserve(tableau_count_);
            for (int destination = 0;
                 destination < tableau_count_;
                 ++destination) {
                if (!can_build(card, state.tableau[destination]) ||
                    destination_already_used(
                        state,
                        destination,
                        used_destinations
                    )) {
                    continue;
                }
                used_destinations.push_back(destination);
                OrderedState next = state;
                --next.waste_size;
                next.tableau[destination]
                    .cards[next.tableau[destination].size++] = card;
                int score =
                    state.tableau[destination].size == 0 ? 180 : 120;
                result.push_back({score, std::move(next)});
            }
        }

        for (int source = 0; source < tableau_count_; ++source) {
            const Pile& source_pile = state.tableau[source];
            if (source_pile.size == 0 ||
                !face_up(source_pile.cards[source_pile.size - 1])) {
                continue;
            }

            uint8_t top = source_pile.cards[source_pile.size - 1];
            if (can_foundation(state, top)) {
                OrderedState next = state;
                --next.tableau[source].size;
                bool reveals = next.tableau[source].size > 0 &&
                    !face_up(
                        next.tableau[source]
                            .cards[next.tableau[source].size - 1]
                    );
                reveal_top(next.tableau[source]);
                ++next.foundations[suit(top)];
                result.push_back({
                    reveals ? 1400 : 550,
                    std::move(next),
                });
            }

            for (int start = 0; start < source_pile.size; ++start) {
                if (!packed_stack(source_pile, start)) {
                    continue;
                }
                uint8_t moving_card = source_pile.cards[start];
                std::vector<int> used_destinations;
                used_destinations.reserve(tableau_count_);
                for (int destination = 0;
                     destination < tableau_count_;
                     ++destination) {
                    if (destination == source ||
                        !can_build(
                            moving_card,
                            state.tableau[destination]
                        ) ||
                        destination_already_used(
                            state,
                            destination,
                            used_destinations
                        )) {
                        continue;
                    }
                    if (state.tableau[destination].size == 0 &&
                        start == 0) {
                        continue;
                    }
                    used_destinations.push_back(destination);

                    OrderedState next = state;
                    Pile& next_source = next.tableau[source];
                    Pile& next_destination =
                        next.tableau[destination];
                    for (int index = start;
                         index < source_pile.size;
                         ++index) {
                        next_destination
                            .cards[next_destination.size++] =
                            source_pile.cards[index];
                    }
                    next_source.size = static_cast<uint8_t>(start);
                    bool reveals = next_source.size > 0 &&
                        !face_up(
                            next_source.cards[next_source.size - 1]
                        );
                    bool empties = next_source.size == 0;
                    reveal_top(next_source);
                    int moved = source_pile.size - start;
                    int score = reveals
                        ? 1300 + moved
                        : (empties ? 700 + moved : 40 + moved);
                    result.push_back({score, std::move(next)});
                }
            }
        }
    }
};

State initial_state(
    const std::vector<uint8_t>& deal,
    int tableau_count
) {
    State state{};
    int cursor = 0;
    for (int column = 0; column < tableau_count; ++column) {
        Pile& pile = state.tableau[column];
        for (int row = 0;
             row <= column && cursor < static_cast<int>(deal.size());
             ++row) {
            uint8_t card = deal[cursor++];
            pile.cards[pile.size++] =
                row == column ? make_face_up(card) : card;
        }
        if (pile.size > 0) {
            pile.cards[pile.size - 1] =
                make_face_up(pile.cards[pile.size - 1]);
        }
    }
    for (; cursor < static_cast<int>(deal.size()); ++cursor) {
        state.reserve[state.reserve_size++] = deal[cursor];
    }
    std::sort(
        state.reserve.begin(),
        state.reserve.begin() + state.reserve_size
    );
    return state;
}

OrderedState initial_ordered_state(
    const std::vector<uint8_t>& deal,
    int tableau_count
) {
    OrderedState state{};
    int cursor = 0;
    for (int column = 0; column < tableau_count; ++column) {
        Pile& pile = state.tableau[column];
        for (int row = 0;
             row <= column && cursor < static_cast<int>(deal.size());
             ++row) {
            uint8_t card = deal[cursor++];
            pile.cards[pile.size++] =
                row == column ? make_face_up(card) : card;
        }
        if (pile.size > 0) {
            pile.cards[pile.size - 1] =
                make_face_up(pile.cards[pile.size - 1]);
        }
    }
    for (int index = static_cast<int>(deal.size()) - 1;
         index >= cursor;
         --index) {
        state.stock[state.stock_size++] = deal[index];
    }
    return state;
}

void generate_tableau_patterns_recursive(
    int n,
    int tableau_cards,
    std::array<int, kSuits>& counts,
    std::vector<uint8_t>& pattern,
    int max_seen,
    std::vector<std::vector<uint8_t>>& output
) {
    if (static_cast<int>(pattern.size()) == tableau_cards) {
        output.push_back(pattern);
        return;
    }
    int maximum_suit = std::min(kSuits - 1, max_seen + 1);
    for (int suit = 0; suit <= maximum_suit; ++suit) {
        if (counts[suit] == n) {
            continue;
        }
        ++counts[suit];
        pattern.push_back(static_cast<uint8_t>(suit));
        generate_tableau_patterns_recursive(
            n,
            tableau_cards,
            counts,
            pattern,
            std::max(max_seen, suit),
            output
        );
        pattern.pop_back();
        --counts[suit];
    }
}

std::vector<std::vector<uint8_t>> generate_tableau_patterns(
    int n,
    int tableau_cards
) {
    std::vector<std::vector<uint8_t>> patterns;
    std::array<int, kSuits> counts{};
    std::vector<uint8_t> pattern;
    generate_tableau_patterns_recursive(
        n,
        tableau_cards,
        counts,
        pattern,
        -1,
        patterns
    );
    return patterns;
}

void generate_partial_orders_recursive(
    int n,
    int count,
    std::vector<uint8_t>& current,
    std::vector<bool>& used,
    std::vector<std::vector<uint8_t>>& output
) {
    if (static_cast<int>(current.size()) == count) {
        output.push_back(current);
        return;
    }
    for (int rank = 0; rank < n; ++rank) {
        if (used[rank]) {
            continue;
        }
        used[rank] = true;
        current.push_back(static_cast<uint8_t>(rank));
        generate_partial_orders_recursive(
            n,
            count,
            current,
            used,
            output
        );
        current.pop_back();
        used[rank] = false;
    }
}

std::vector<std::vector<uint8_t>> partial_rank_orders(
    int n,
    int count
) {
    std::vector<std::vector<uint8_t>> output;
    std::vector<uint8_t> current;
    std::vector<bool> used(n, false);
    generate_partial_orders_recursive(n, count, current, used, output);
    return output;
}

struct PatternWork {
    std::vector<uint8_t> pattern;
    std::array<int, kSuits> counts{};
    uint64_t combinations = 0;
    uint64_t byte_offset = 0;
};

bool read_exact(int file, void* buffer, size_t size, off_t offset) {
    uint8_t* destination = static_cast<uint8_t*>(buffer);
    size_t total = 0;
    while (total < size) {
        ssize_t count = pread(
            file,
            destination + total,
            size - total,
            offset + static_cast<off_t>(total)
        );
        if (count <= 0) {
            return false;
        }
        total += static_cast<size_t>(count);
    }
    return true;
}

bool write_exact(
    int file,
    const void* buffer,
    size_t size,
    off_t offset
) {
    const uint8_t* source = static_cast<const uint8_t*>(buffer);
    size_t total = 0;
    while (total < size) {
        ssize_t count = pwrite(
            file,
            source + total,
            size - total,
            offset + static_cast<off_t>(total)
        );
        if (count <= 0) {
            return false;
        }
        total += static_cast<size_t>(count);
    }
    return true;
}

int open_sized_file(const std::string& path, off_t size) {
    int file = open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (file < 0) {
        return -1;
    }
    struct stat info{};
    if (fstat(file, &info) != 0) {
        close(file);
        return -1;
    }
    if (info.st_size == 0) {
        if (ftruncate(file, size) != 0) {
            close(file);
            return -1;
        }
    } else if (info.st_size != size) {
        std::cerr << "unexpected size for " << path << "\n";
        close(file);
        return -1;
    }
    return file;
}

uint64_t popcount_file(const std::string& path, uint64_t bytes) {
    int file = open(path.c_str(), O_RDONLY);
    if (file < 0) {
        throw std::runtime_error("could not open outcome file");
    }
    std::vector<uint8_t> buffer(4 * 1024 * 1024);
    uint64_t total = 0;
    uint64_t offset = 0;
    while (offset < bytes) {
        size_t chunk = static_cast<size_t>(
            std::min<uint64_t>(buffer.size(), bytes - offset)
        );
        if (!read_exact(file, buffer.data(), chunk, offset)) {
            close(file);
            throw std::runtime_error("could not read outcome file");
        }
        for (size_t index = 0; index < chunk; ++index) {
            total += static_cast<uint64_t>(
                __builtin_popcount(buffer[index])
            );
        }
        offset += chunk;
    }
    close(file);
    return total;
}

uint64_t sum_u16_file(const std::string& path, uint64_t values) {
    int file = open(path.c_str(), O_RDONLY);
    if (file < 0) {
        throw std::runtime_error("could not open ordered-count file");
    }
    std::vector<uint16_t> buffer(1024 * 1024);
    uint64_t total = 0;
    uint64_t offset = 0;
    while (offset < values) {
        size_t count = static_cast<size_t>(
            std::min<uint64_t>(buffer.size(), values - offset)
        );
        size_t bytes = count * sizeof(uint16_t);
        if (!read_exact(
                file,
                buffer.data(),
                bytes,
                static_cast<off_t>(offset * sizeof(uint16_t)))) {
            close(file);
            throw std::runtime_error(
                "could not read ordered-count file"
            );
        }
        for (size_t index = 0; index < count; ++index) {
            total += buffer[index];
        }
        offset += count;
    }
    close(file);
    return total;
}

std::vector<uint8_t> parse_deal(const std::string& encoded) {
    std::vector<uint8_t> deal;
    std::stringstream stream(encoded);
    std::string item;
    while (std::getline(stream, item, ',')) {
        deal.push_back(static_cast<uint8_t>(std::stoi(item)));
    }
    return deal;
}

struct StockOrder {
    std::array<uint8_t, kMaxCards> cards{};
    uint8_t size = 0;
};

int passes_to_emit(const StockOrder& relative_draw_order) {
    std::vector<uint8_t> remaining(
        relative_draw_order.cards.begin(),
        relative_draw_order.cards.begin() + relative_draw_order.size
    );
    int target = 0;
    int passes = 0;
    while (!remaining.empty()) {
        ++passes;
        std::vector<uint8_t> waste;
        waste.reserve(remaining.size());
        for (uint8_t card : remaining) {
            waste.push_back(card);
            while (!waste.empty() && waste.back() == target) {
                waste.pop_back();
                ++target;
            }
        }
        if (waste.size() == remaining.size()) {
            throw std::runtime_error(
                "stock pass failed to emit its next target"
            );
        }
        remaining = std::move(waste);
    }
    return passes;
}

std::vector<StockOrder> nonreplayable_relative_orders(
    int card_count,
    int available_passes
) {
    std::vector<uint8_t> order(card_count);
    std::iota(order.begin(), order.end(), 0);
    std::vector<StockOrder> output;
    do {
        StockOrder candidate;
        candidate.size = static_cast<uint8_t>(card_count);
        std::copy(
            order.begin(),
            order.end(),
            candidate.cards.begin()
        );
        if (passes_to_emit(candidate) > available_passes) {
            output.push_back(candidate);
        }
    } while (std::next_permutation(order.begin(), order.end()));
    return output;
}

StockOrder instantiate_relative_order(
    const std::vector<uint8_t>& target_order,
    const StockOrder& relative_order
) {
    StockOrder actual;
    actual.size = relative_order.size;
    for (int index = 0; index < relative_order.size; ++index) {
        actual.cards[index] =
            target_order[relative_order.cards[index]];
    }
    return actual;
}

bool replayable_with_target(
    const std::vector<uint8_t>& target_order,
    const StockOrder& actual_draw_order,
    int available_passes
) {
    std::array<uint8_t, kMaxCards> target_index{};
    for (size_t index = 0; index < target_order.size(); ++index) {
        target_index[target_order[index]] =
            static_cast<uint8_t>(index);
    }
    StockOrder relative;
    relative.size = actual_draw_order.size;
    for (int index = 0; index < actual_draw_order.size; ++index) {
        relative.cards[index] =
            target_index[actual_draw_order.cards[index]];
    }
    return passes_to_emit(relative) <= available_passes;
}

struct Options {
    int n = 7;
    int threads = 1;
    uint64_t max_patterns = 0;
    uint64_t benchmark_deals = 0;
    uint64_t benchmark_seed = 20260716;
    std::string output;
    std::string completion;
    std::string deal;
    std::string deals_file;
    std::string outcomes_file;
    std::string stock_mode = "reserve";
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        std::string argument = argv[index];
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        std::string value = argv[++index];
        if (argument == "--n") {
            options.n = std::stoi(value);
        } else if (argument == "--threads") {
            options.threads = std::stoi(value);
        } else if (argument == "--max-patterns") {
            options.max_patterns = std::stoull(value);
        } else if (argument == "--benchmark-deals") {
            options.benchmark_deals = std::stoull(value);
        } else if (argument == "--benchmark-seed") {
            options.benchmark_seed = std::stoull(value);
        } else if (argument == "--output") {
            options.output = value;
        } else if (argument == "--completion") {
            options.completion = value;
        } else if (argument == "--deal") {
            options.deal = value;
        } else if (argument == "--deals-file") {
            options.deals_file = value;
        } else if (argument == "--outcomes-file") {
            options.outcomes_file = value;
        } else if (argument == "--stock-mode") {
            options.stock_mode = value;
        } else {
            throw std::runtime_error("unknown argument " + argument);
        }
    }
    if (options.n < 1 || options.n > 13 || options.threads < 1) {
        throw std::runtime_error("require 1 <= n <= 13 and threads >= 1");
    }
    if (!options.deals_file.empty() && options.outcomes_file.empty()) {
        throw std::runtime_error(
            "--deals-file requires --outcomes-file"
        );
    }
    if (options.stock_mode != "reserve" &&
        options.stock_mode != "ordered") {
        throw std::runtime_error(
            "--stock-mode must be reserve or ordered"
        );
    }
    if (options.deal.empty() && options.deals_file.empty() &&
        options.benchmark_deals == 0 &&
        (options.output.empty() || options.completion.empty())) {
        throw std::runtime_error(
            "checkpoint paths are required for an exhaustive run"
        );
    }
    int stock_cards = stock_card_count(options.n);
    if (stock_cards < 0) {
        throw std::runtime_error("default tableau exceeds the deck");
    }
    if (options.deal.empty() && options.deals_file.empty() &&
        options.stock_mode == "reserve" && stock_cards > 6) {
        throw std::runtime_error(
            "the four-pass stock-order theorem requires at most six stock cards"
        );
    }
    return options;
}

int run_single_deal(const Options& options) {
    std::vector<uint8_t> deal = parse_deal(options.deal);
    int deck_size = kSuits * options.n;
    std::vector<uint8_t> sorted = deal;
    std::sort(sorted.begin(), sorted.end());
    std::vector<uint8_t> expected(deck_size);
    std::iota(expected.begin(), expected.end(), 0);
    if (sorted != expected) {
        throw std::runtime_error(
            "--deal must contain each card id exactly once"
        );
    }
    int tableau_count = default_tableau_count(options.n);
    bool solvable = false;
    uint64_t expanded = 0;
    if (options.stock_mode == "ordered") {
        OrderedSolver solver(options.n, tableau_count);
        solvable = solver.is_solvable(
            initial_ordered_state(deal, tableau_count)
        );
        expanded = solver.expanded_positions;
    } else {
        Solver solver(options.n, tableau_count);
        solvable = solver.is_solvable(
            initial_state(deal, tableau_count)
        );
        expanded = solver.expanded_positions;
    }
    std::cout << "solvable " << (solvable ? 1 : 0) << "\n";
    std::cout << "expanded_positions " << expanded << "\n";
    return 0;
}

void validate_deal(
    const std::vector<uint8_t>& deal,
    int deck_size
) {
    std::vector<uint8_t> sorted = deal;
    std::sort(sorted.begin(), sorted.end());
    std::vector<uint8_t> expected(deck_size);
    std::iota(expected.begin(), expected.end(), 0);
    if (sorted != expected) {
        throw std::runtime_error(
            "each deal must contain every card id exactly once"
        );
    }
}

int run_deals_file(const Options& options) {
    std::ifstream input(options.deals_file);
    std::ofstream output(
        options.outcomes_file,
        std::ios::binary | std::ios::trunc
    );
    if (!input || !output) {
        throw std::runtime_error(
            "could not open validation deal or outcome file"
        );
    }
    int deck_size = kSuits * options.n;
    int tableau_count = default_tableau_count(options.n);
    uint64_t deals = 0;
    uint64_t solvable = 0;
    uint64_t expanded = 0;
    uint64_t normalized = 0;
    std::string line;
    if (options.stock_mode == "ordered") {
        OrderedSolver solver(options.n, tableau_count);
        while (std::getline(input, line)) {
            if (line.empty()) {
                continue;
            }
            std::vector<uint8_t> deal = parse_deal(line);
            validate_deal(deal, deck_size);
            uint8_t outcome = solver.is_solvable(
                initial_ordered_state(deal, tableau_count)
            ) ? 1 : 0;
            output.write(
                reinterpret_cast<const char*>(&outcome),
                sizeof(outcome)
            );
            ++deals;
            solvable += outcome;
        }
        expanded = solver.expanded_positions;
        normalized = solver.normalized_foundation_moves;
    } else {
        Solver solver(options.n, tableau_count);
        while (std::getline(input, line)) {
            if (line.empty()) {
                continue;
            }
            std::vector<uint8_t> deal = parse_deal(line);
            validate_deal(deal, deck_size);
            uint8_t outcome = solver.is_solvable(
                initial_state(deal, tableau_count)
            ) ? 1 : 0;
            output.write(
                reinterpret_cast<const char*>(&outcome),
                sizeof(outcome)
            );
            ++deals;
            solvable += outcome;
        }
        expanded = solver.expanded_positions;
        normalized = solver.normalized_foundation_moves;
    }
    if (!output) {
        throw std::runtime_error(
            "could not write validation outcomes"
        );
    }
    std::cout << "verified_deals " << deals << "\n";
    std::cout << "solvable_deals " << solvable << "\n";
    std::cout << "expanded_positions " << expanded << "\n";
    std::cout << "normalized_foundation_moves "
              << normalized << "\n";
    return 0;
}

int run_benchmark(const Options& options) {
    int tableau_count = default_tableau_count(options.n);
    int tableau_cards = tableau_count * (tableau_count + 1) / 2;
    int deck_size = kSuits * options.n;
    std::atomic<uint64_t> next_deal{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> solvable{0};
    std::atomic<uint64_t> expanded{0};
    std::atomic<uint64_t> normalized{0};
    std::atomic<uint64_t> reserve_wins{0};
    std::atomic<uint64_t> reserve_losses{0};
    std::atomic<uint64_t> replay_certified{0};
    std::atomic<uint64_t> ordered_exact_checks{0};
    std::atomic<uint64_t> ordered_exact_wins{0};
    std::mutex progress_mutex;
    auto started = std::chrono::steady_clock::now();
    uint64_t progress_interval = std::max<uint64_t>(
        1,
        options.benchmark_deals / 100
    );

    auto record_outcome = [&](bool outcome) {
        solvable.fetch_add(outcome);
        uint64_t current = completed.fetch_add(1) + 1;
        if (current % progress_interval != 0 &&
            current != options.benchmark_deals) {
            return;
        }
        double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        double rate = current / std::max(seconds, 0.001);
        double remaining =
            (options.benchmark_deals - current) /
            std::max(rate, 0.001);
        std::lock_guard<std::mutex> lock(progress_mutex);
        std::cout
            << "benchmark_progress " << current << "/"
            << options.benchmark_deals
            << " solvable " << solvable.load()
            << " rate " << rate << "/s"
            << " eta " << remaining / 60.0 << "m\n"
            << std::flush;
    };

    auto worker = [&]() {
        std::vector<uint8_t> deal(deck_size);
        if (options.stock_mode == "ordered") {
            constexpr int available_passes = 4;
            constexpr int certificate_variations = 4;
            Solver reserve_solver(options.n, tableau_count);
            OrderedSolver solver(options.n, tableau_count);
            while (true) {
                uint64_t index = next_deal.fetch_add(1);
                if (index >= options.benchmark_deals) {
                    break;
                }
                std::iota(deal.begin(), deal.end(), 0);
                std::mt19937_64 random(
                    options.benchmark_seed +
                    index * 0x9e3779b97f4a7c15ULL
                );
                std::shuffle(deal.begin(), deal.end(), random);
                StockOrder actual_stock;
                actual_stock.size = static_cast<uint8_t>(
                    deck_size - tableau_cards
                );
                for (int stock_index = 0;
                     stock_index < actual_stock.size;
                     ++stock_index) {
                    actual_stock.cards[stock_index] =
                        deal[tableau_cards + stock_index];
                }

                State reserve_state =
                    initial_state(deal, tableau_count);
                std::vector<uint8_t> target;
                bool reserve_solvable =
                    reserve_solver.find_solution(
                        reserve_state,
                        target,
                        0
                    );
                if (!reserve_solvable) {
                    reserve_losses.fetch_add(1);
                    record_outcome(false);
                    continue;
                }
                reserve_wins.fetch_add(1);
                if (target.size() != actual_stock.size) {
                    throw std::runtime_error(
                        "benchmark certificate omitted a stock card"
                    );
                }

                bool certified = replayable_with_target(
                    target,
                    actual_stock,
                    available_passes
                );
                for (int variation = 1;
                     variation < certificate_variations &&
                     !certified;
                     ++variation) {
                    target.clear();
                    if (!reserve_solver.find_solution(
                            reserve_state,
                            target,
                            variation)) {
                        throw std::runtime_error(
                            "alternate benchmark certificate lost a win"
                        );
                    }
                    if (target.size() != actual_stock.size) {
                        throw std::runtime_error(
                            "alternate benchmark certificate "
                            "omitted a stock card"
                        );
                    }
                    certified = replayable_with_target(
                        target,
                        actual_stock,
                        available_passes
                    );
                }
                if (certified) {
                    replay_certified.fetch_add(1);
                    record_outcome(true);
                    continue;
                }

                ordered_exact_checks.fetch_add(1);
                bool outcome = solver.is_solvable(
                    initial_ordered_state(deal, tableau_count)
                );
                ordered_exact_wins.fetch_add(outcome);
                record_outcome(outcome);
            }
            expanded.fetch_add(
                reserve_solver.expanded_positions +
                solver.expanded_positions
            );
            normalized.fetch_add(
                reserve_solver.normalized_foundation_moves +
                solver.normalized_foundation_moves
            );
        } else {
            Solver solver(options.n, tableau_count);
            while (true) {
                uint64_t index = next_deal.fetch_add(1);
                if (index >= options.benchmark_deals) {
                    break;
                }
                std::iota(deal.begin(), deal.end(), 0);
                std::mt19937_64 random(
                    options.benchmark_seed +
                    index * 0x9e3779b97f4a7c15ULL
                );
                std::shuffle(deal.begin(), deal.end(), random);
                std::sort(
                    deal.begin() + tableau_cards,
                    deal.end()
                );
                record_outcome(
                    solver.is_solvable(
                        initial_state(deal, tableau_count)
                    )
                );
            }
            expanded.fetch_add(solver.expanded_positions);
            normalized.fetch_add(
                solver.normalized_foundation_moves
            );
        }
    };

    std::vector<std::thread> threads;
    for (int index = 0; index < options.threads; ++index) {
        threads.emplace_back(worker);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    std::cout << "n " << options.n << "\n";
    std::cout << "k " << kSuits << "\n";
    std::cout << "t " << tableau_count << "\n";
    std::cout << "tableau_cards " << tableau_cards << "\n";
    std::cout << "stock_cards "
              << deck_size - tableau_cards << "\n";
    std::cout << "stock_mode " << options.stock_mode << "\n";
    std::cout << "benchmark_seed "
              << options.benchmark_seed << "\n";
    std::cout << "benchmark_deals "
              << options.benchmark_deals << "\n";
    std::cout << "solvable_deals " << solvable.load() << "\n";
    std::cout << "expanded_positions " << expanded.load() << "\n";
    std::cout << "normalized_foundation_moves " << normalized.load() << "\n";
    if (options.stock_mode == "ordered") {
        std::cout << "reserve_wins "
                  << reserve_wins.load() << "\n";
        std::cout << "reserve_losses "
                  << reserve_losses.load() << "\n";
        std::cout << "replay_certified_deals "
                  << replay_certified.load() << "\n";
        std::cout << "ordered_exact_checks "
                  << ordered_exact_checks.load() << "\n";
        std::cout << "ordered_exact_wins "
                  << ordered_exact_wins.load() << "\n";
    }
    std::cout << "elapsed_seconds " << elapsed << "\n";
    std::cout << "deals_per_second "
              << options.benchmark_deals / elapsed << "\n";
    return 0;
}

int run_ordered_exhaustive(const Options& options) {
    constexpr int available_passes = 4;
    int n = options.n;
    int tableau_count = default_tableau_count(n);
    int tableau_cards = tableau_count * (tableau_count + 1) / 2;
    int deck_size = kSuits * n;
    int stock_cards = deck_size - tableau_cards;
    uint64_t stock_permutations = factorial(stock_cards);
    if (stock_permutations > UINT16_MAX) {
        throw std::runtime_error(
            "ordered-count output supports at most 65,535 stock orders"
        );
    }
    std::vector<StockOrder> bad_relative_orders =
        nonreplayable_relative_orders(
            stock_cards,
            available_passes
        );
    auto patterns = generate_tableau_patterns(n, tableau_cards);

    std::vector<std::vector<std::vector<uint8_t>>> rank_orders(n + 1);
    for (int count = 0; count <= n; ++count) {
        rank_orders[count] = partial_rank_orders(n, count);
    }

    std::vector<PatternWork> work;
    work.reserve(patterns.size());
    uint64_t total_tableau_deals = 0;
    for (std::vector<uint8_t>& pattern : patterns) {
        PatternWork item;
        item.pattern = std::move(pattern);
        for (uint8_t suit : item.pattern) {
            ++item.counts[suit];
        }
        item.combinations = 1;
        for (int suit = 0; suit < kSuits; ++suit) {
            item.combinations *=
                rank_orders[item.counts[suit]].size();
        }
        item.byte_offset =
            total_tableau_deals * sizeof(uint16_t);
        total_tableau_deals += item.combinations;
        work.push_back(std::move(item));
    }

    uint64_t expected_tableau_deals =
        factorial(deck_size) /
        factorial(stock_cards) /
        factorial(kSuits);
    if (total_tableau_deals != expected_tableau_deals) {
        throw std::runtime_error(
            "ordered reduced tableau count mismatch"
        );
    }
    uint64_t total_ordered_deals =
        total_tableau_deals * stock_permutations;
    if (total_ordered_deals !=
        factorial(deck_size) / factorial(kSuits)) {
        throw std::runtime_error(
            "ordered reduced deal count mismatch"
        );
    }

    int output_file = open_sized_file(
        options.output,
        static_cast<off_t>(
            total_tableau_deals * sizeof(uint16_t)
        )
    );
    int completion_file = open_sized_file(
        options.completion,
        static_cast<off_t>(work.size())
    );
    if (output_file < 0 || completion_file < 0) {
        throw std::runtime_error(
            "could not open ordered checkpoint files"
        );
    }

    std::vector<uint8_t> completion(work.size());
    if (!read_exact(
            completion_file,
            completion.data(),
            completion.size(),
            0)) {
        close(output_file);
        close(completion_file);
        throw std::runtime_error(
            "could not read ordered completion checkpoint"
        );
    }

    uint64_t already_patterns =
        std::count(completion.begin(), completion.end(), 1);
    uint64_t already_tableaus = 0;
    for (size_t index = 0; index < work.size(); ++index) {
        if (completion[index] == 1) {
            already_tableaus += work[index].combinations;
        }
    }

    std::atomic<uint64_t> next_pattern{0};
    std::atomic<uint64_t> claimed{0};
    std::atomic<uint64_t> patterns_done{already_patterns};
    std::atomic<uint64_t> tableaus_done{already_tableaus};
    std::atomic<uint64_t> reserve_wins{0};
    std::atomic<uint64_t> reserve_losses{0};
    std::atomic<uint64_t> replay_certified{0};
    std::atomic<uint64_t> exact_checks{0};
    std::atomic<uint64_t> exact_wins{0};
    std::atomic<uint64_t> reserve_expanded{0};
    std::atomic<uint64_t> ordered_expanded{0};
    std::atomic<uint64_t> normalized{0};
    std::mutex output_mutex;
    auto started = std::chrono::steady_clock::now();

    auto worker = [&]() {
        Solver reserve_solver(n, tableau_count);
        OrderedSolver ordered_solver(n, tableau_count);
        std::vector<uint8_t> deal(deck_size);
        std::vector<StockOrder> unresolved;
        unresolved.reserve(bad_relative_orders.size());
        while (true) {
            uint64_t pattern_index = next_pattern.fetch_add(1);
            if (pattern_index >= work.size()) {
                break;
            }
            if (completion[pattern_index] == 1) {
                continue;
            }
            uint64_t claim = claimed.fetch_add(1);
            if (options.max_patterns > 0 &&
                claim >= options.max_patterns) {
                break;
            }

            const PatternWork& item = work[pattern_index];
            std::vector<uint16_t> solvable_counts(
                item.combinations,
                0
            );
            uint64_t pattern_reserve_wins = 0;
            uint64_t pattern_reserve_losses = 0;
            uint64_t pattern_certified = 0;
            uint64_t pattern_exact_checks = 0;
            uint64_t pattern_exact_wins = 0;

            for (uint64_t combination = 0;
                 combination < item.combinations;
                 ++combination) {
                uint64_t remainder = combination;
                std::array<size_t, kSuits> order_index{};
                for (int suit = kSuits - 1; suit >= 0; --suit) {
                    size_t order_count =
                        rank_orders[item.counts[suit]].size();
                    order_index[suit] = remainder % order_count;
                    remainder /= order_count;
                }

                std::array<size_t, kSuits> occurrence{};
                std::array<bool, kMaxCards> used{};
                for (int position = 0;
                     position < tableau_cards;
                     ++position) {
                    int suit = item.pattern[position];
                    uint8_t rank_index =
                        rank_orders[item.counts[suit]]
                            [order_index[suit]]
                            [occurrence[suit]++];
                    uint8_t card = static_cast<uint8_t>(
                        suit * n + rank_index
                    );
                    deal[position] = card;
                    used[card] = true;
                }
                int stock_position = tableau_cards;
                for (int card = 0; card < deck_size; ++card) {
                    if (!used[card]) {
                        deal[stock_position++] =
                            static_cast<uint8_t>(card);
                    }
                }

                State reserve_state =
                    initial_state(deal, tableau_count);
                std::vector<uint8_t> first_target;
                if (!reserve_solver.find_solution(
                        reserve_state,
                        first_target,
                        0)) {
                    ++pattern_reserve_losses;
                    continue;
                }
                ++pattern_reserve_wins;
                if (static_cast<int>(first_target.size()) !=
                    stock_cards) {
                    throw std::runtime_error(
                        "reserve certificate omitted a stock card"
                    );
                }

                uint64_t solved_orders =
                    stock_permutations -
                    bad_relative_orders.size();
                pattern_certified += solved_orders;
                unresolved.clear();
                for (const StockOrder& relative :
                     bad_relative_orders) {
                    unresolved.push_back(
                        instantiate_relative_order(
                            first_target,
                            relative
                        )
                    );
                }

                std::vector<std::vector<uint8_t>> targets;
                targets.push_back(first_target);
                for (int variation = 1;
                     variation < 4 && !unresolved.empty();
                     ++variation) {
                    std::vector<uint8_t> target;
                    if (!reserve_solver.find_solution(
                            reserve_state,
                            target,
                            variation)) {
                        throw std::runtime_error(
                            "alternate reserve search lost a known win"
                        );
                    }
                    if (static_cast<int>(target.size()) !=
                        stock_cards) {
                        throw std::runtime_error(
                            "alternate certificate omitted a stock card"
                        );
                    }
                    if (std::find(
                            targets.begin(),
                            targets.end(),
                            target) != targets.end()) {
                        continue;
                    }
                    targets.push_back(target);
                    auto first_uncovered = std::remove_if(
                        unresolved.begin(),
                        unresolved.end(),
                        [&target](const StockOrder& actual) {
                            return replayable_with_target(
                                target,
                                actual,
                                available_passes
                            );
                        }
                    );
                    uint64_t newly_certified =
                        static_cast<uint64_t>(
                            unresolved.end() - first_uncovered
                        );
                    solved_orders += newly_certified;
                    pattern_certified += newly_certified;
                    unresolved.erase(
                        first_uncovered,
                        unresolved.end()
                    );
                }

                for (const StockOrder& actual : unresolved) {
                    for (int index = 0; index < stock_cards; ++index) {
                        deal[tableau_cards + index] =
                            actual.cards[index];
                    }
                    ++pattern_exact_checks;
                    if (ordered_solver.is_solvable(
                            initial_ordered_state(
                                deal,
                                tableau_count
                            ))) {
                        ++solved_orders;
                        ++pattern_exact_wins;
                    }
                }
                solvable_counts[combination] =
                    static_cast<uint16_t>(solved_orders);
            }

            if (!write_exact(
                    output_file,
                    solvable_counts.data(),
                    solvable_counts.size() * sizeof(uint16_t),
                    static_cast<off_t>(item.byte_offset))) {
                throw std::runtime_error(
                    "could not write ordered-count checkpoint"
                );
            }
            uint8_t complete = 1;
            if (!write_exact(
                    completion_file,
                    &complete,
                    1,
                    static_cast<off_t>(pattern_index))) {
                throw std::runtime_error(
                    "could not write ordered completion checkpoint"
                );
            }
            completion[pattern_index] = 1;

            reserve_wins.fetch_add(pattern_reserve_wins);
            reserve_losses.fetch_add(pattern_reserve_losses);
            replay_certified.fetch_add(pattern_certified);
            exact_checks.fetch_add(pattern_exact_checks);
            exact_wins.fetch_add(pattern_exact_wins);
            uint64_t current_tableaus =
                tableaus_done.fetch_add(item.combinations) +
                item.combinations;
            uint64_t current_patterns =
                patterns_done.fetch_add(1) + 1;
            double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started
            ).count();
            uint64_t new_tableaus =
                current_tableaus - already_tableaus;
            double rate =
                new_tableaus / std::max(seconds, 0.001);
            double remaining =
                (total_tableau_deals - current_tableaus) /
                std::max(rate, 0.001);
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout
                << "tableau_patterns "
                << current_patterns << "/" << work.size()
                << " tableau_deals " << current_tableaus << "/"
                << total_tableau_deals
                << " exact_checks " << exact_checks.load()
                << " tableau_rate " << rate << "/s"
                << " eta " << remaining / 60.0 << "m\n"
                << std::flush;
        }
        reserve_expanded.fetch_add(
            reserve_solver.expanded_positions
        );
        ordered_expanded.fetch_add(
            ordered_solver.expanded_positions
        );
        normalized.fetch_add(
            reserve_solver.normalized_foundation_moves +
            ordered_solver.normalized_foundation_moves
        );
    };

    std::vector<std::thread> threads;
    for (int index = 0; index < options.threads; ++index) {
        threads.emplace_back(worker);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    fsync(output_file);
    fsync(completion_file);
    close(output_file);
    close(completion_file);

    uint64_t completed_patterns =
        std::count(completion.begin(), completion.end(), 1);
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    std::cout << "completed_tableau_patterns "
              << completed_patterns << "/" << work.size() << "\n";
    std::cout << "canonical_tableau_deals "
              << total_tableau_deals << "\n";
    std::cout << "stock_permutations "
              << stock_permutations << "\n";
    std::cout << "canonical_ordered_deals "
              << total_ordered_deals << "\n";
    std::cout << "nonreplayable_per_certificate "
              << bad_relative_orders.size() << "\n";
    std::cout << "classified_tableau_deals "
              << tableaus_done.load() << "\n";
    std::cout << "reserve_wins " << reserve_wins.load() << "\n";
    std::cout << "reserve_losses " << reserve_losses.load() << "\n";
    std::cout << "replay_certified_orders "
              << replay_certified.load() << "\n";
    std::cout << "ordered_exact_checks "
              << exact_checks.load() << "\n";
    std::cout << "ordered_exact_wins "
              << exact_wins.load() << "\n";
    std::cout << "reserve_expanded_positions "
              << reserve_expanded.load() << "\n";
    std::cout << "ordered_expanded_positions "
              << ordered_expanded.load() << "\n";
    std::cout << "normalized_foundation_moves "
              << normalized.load() << "\n";
    std::cout << "elapsed_seconds " << elapsed << "\n";

    if (completed_patterns == work.size()) {
        uint64_t solvable = sum_u16_file(
            options.output,
            total_tableau_deals
        );
        std::cout << "solvable_ordered_deals "
                  << solvable << "\n";
        std::cout << "unsolvable_ordered_deals "
                  << total_ordered_deals - solvable << "\n";
        std::cout << "solvability_rate "
                  << static_cast<double>(solvable) /
                        total_ordered_deals
                  << "\n";
    }
    return 0;
}

int run_exhaustive(const Options& options) {
    int n = options.n;
    int tableau_count = default_tableau_count(n);
    int tableau_cards = tableau_count * (tableau_count + 1) / 2;
    int deck_size = kSuits * n;
    int stock_cards = deck_size - tableau_cards;
    auto patterns = generate_tableau_patterns(n, tableau_cards);

    std::vector<std::vector<std::vector<uint8_t>>> rank_orders(n + 1);
    for (int count = 0; count <= n; ++count) {
        rank_orders[count] = partial_rank_orders(n, count);
    }

    std::vector<PatternWork> work;
    work.reserve(patterns.size());
    uint64_t total_deals = 0;
    uint64_t total_bytes = 0;
    for (std::vector<uint8_t>& pattern : patterns) {
        PatternWork item;
        item.pattern = std::move(pattern);
        for (uint8_t suit : item.pattern) {
            ++item.counts[suit];
        }
        item.combinations = 1;
        for (int suit = 0; suit < kSuits; ++suit) {
            item.combinations *= rank_orders[item.counts[suit]].size();
        }
        item.byte_offset = total_bytes;
        total_deals += item.combinations;
        total_bytes += (item.combinations + 7) / 8;
        work.push_back(std::move(item));
    }

    uint64_t expected_deals =
        factorial(deck_size) / factorial(stock_cards) / factorial(kSuits);
    if (total_deals != expected_deals) {
        throw std::runtime_error("reduced deal count mismatch");
    }

    int output_file = open_sized_file(
        options.output,
        static_cast<off_t>(total_bytes)
    );
    int completion_file = open_sized_file(
        options.completion,
        static_cast<off_t>(work.size())
    );
    if (output_file < 0 || completion_file < 0) {
        throw std::runtime_error("could not open checkpoint files");
    }

    std::vector<uint8_t> completion(work.size());
    if (!read_exact(
            completion_file,
            completion.data(),
            completion.size(),
            0)) {
        close(output_file);
        close(completion_file);
        throw std::runtime_error("could not read completion checkpoint");
    }

    uint64_t already_patterns =
        std::count(completion.begin(), completion.end(), 1);
    uint64_t already_deals = 0;
    for (size_t index = 0; index < work.size(); ++index) {
        if (completion[index] == 1) {
            already_deals += work[index].combinations;
        }
    }

    std::atomic<uint64_t> next_pattern{0};
    std::atomic<uint64_t> claimed{0};
    std::atomic<uint64_t> patterns_done{already_patterns};
    std::atomic<uint64_t> deals_done{already_deals};
    std::atomic<uint64_t> expanded{0};
    std::atomic<uint64_t> normalized{0};
    std::atomic<uint64_t> solvable_cache_hits{0};
    std::atomic<uint64_t> unsolvable_cache_hits{0};
    std::mutex output_mutex;
    auto started = std::chrono::steady_clock::now();

    auto worker = [&]() {
        Solver solver(n, tableau_count);
        std::vector<uint8_t> deal(deck_size);
        while (true) {
            uint64_t pattern_index = next_pattern.fetch_add(1);
            if (pattern_index >= work.size()) {
                break;
            }
            if (completion[pattern_index] == 1) {
                continue;
            }
            uint64_t claim = claimed.fetch_add(1);
            if (options.max_patterns > 0 &&
                claim >= options.max_patterns) {
                break;
            }

            const PatternWork& item = work[pattern_index];
            std::vector<uint8_t> outcomes(
                (item.combinations + 7) / 8,
                0
            );
            for (uint64_t combination = 0;
                 combination < item.combinations;
                 ++combination) {
                uint64_t remainder = combination;
                std::array<size_t, kSuits> order_index{};
                for (int suit = kSuits - 1; suit >= 0; --suit) {
                    size_t order_count =
                        rank_orders[item.counts[suit]].size();
                    order_index[suit] = remainder % order_count;
                    remainder /= order_count;
                }

                std::array<size_t, kSuits> occurrence{};
                std::array<bool, kMaxCards> used{};
                for (int position = 0;
                     position < tableau_cards;
                     ++position) {
                    int suit = item.pattern[position];
                    uint8_t rank_index =
                        rank_orders[item.counts[suit]]
                            [order_index[suit]]
                            [occurrence[suit]++];
                    uint8_t card = static_cast<uint8_t>(
                        suit * n + rank_index
                    );
                    deal[position] = card;
                    used[card] = true;
                }
                int stock_position = tableau_cards;
                for (int card = 0; card < deck_size; ++card) {
                    if (!used[card]) {
                        deal[stock_position++] =
                            static_cast<uint8_t>(card);
                    }
                }

                if (solver.is_solvable(
                        initial_state(deal, tableau_count))) {
                    outcomes[combination / 8] |=
                        static_cast<uint8_t>(
                            1U << (combination % 8)
                        );
                }
            }

            if (!write_exact(
                    output_file,
                    outcomes.data(),
                    outcomes.size(),
                    static_cast<off_t>(item.byte_offset))) {
                throw std::runtime_error(
                    "could not write outcome checkpoint"
                );
            }
            uint8_t complete = 1;
            if (!write_exact(
                    completion_file,
                    &complete,
                    1,
                    static_cast<off_t>(pattern_index))) {
                throw std::runtime_error(
                    "could not write completion checkpoint"
                );
            }
            completion[pattern_index] = 1;
            uint64_t current_deals =
                deals_done.fetch_add(item.combinations) +
                item.combinations;
            uint64_t current_patterns =
                patterns_done.fetch_add(1) + 1;
            if (current_patterns % 10 == 0 ||
                current_patterns == work.size()) {
                double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started
                ).count();
                uint64_t new_deals = current_deals - already_deals;
                double rate = new_deals / std::max(seconds, 0.001);
                double remaining =
                    (total_deals - current_deals) /
                    std::max(rate, 0.001);
                std::lock_guard<std::mutex> lock(output_mutex);
                std::cout
                    << "tableau_patterns "
                    << current_patterns << "/" << work.size()
                    << " deals " << current_deals << "/"
                    << total_deals
                    << " deal_rate " << rate << "/s"
                    << " eta " << remaining / 60.0 << "m\n"
                    << std::flush;
            }
        }
        expanded.fetch_add(solver.expanded_positions);
        normalized.fetch_add(solver.normalized_foundation_moves);
        solvable_cache_hits.fetch_add(solver.solvable_cache_hits);
        unsolvable_cache_hits.fetch_add(solver.unsolvable_cache_hits);
    };

    std::vector<std::thread> threads;
    for (int index = 0; index < options.threads; ++index) {
        threads.emplace_back(worker);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    fsync(output_file);
    fsync(completion_file);
    close(output_file);
    close(completion_file);

    uint64_t completed_patterns =
        std::count(completion.begin(), completion.end(), 1);
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    std::cout << "completed_tableau_patterns "
              << completed_patterns << "/" << work.size() << "\n";
    std::cout << "canonical_tableau_deals " << total_deals << "\n";
    std::cout << "outcome_bytes " << total_bytes << "\n";
    std::cout << "classified_tableau_deals " << deals_done.load() << "\n";
    std::cout << "expanded_positions " << expanded.load() << "\n";
    std::cout << "normalized_foundation_moves "
              << normalized.load() << "\n";
    std::cout << "solvable_cache_hits "
              << solvable_cache_hits.load() << "\n";
    std::cout << "unsolvable_cache_hits "
              << unsolvable_cache_hits.load() << "\n";
    std::cout << "elapsed_seconds " << elapsed << "\n";

    if (completed_patterns == work.size()) {
        uint64_t solvable = popcount_file(options.output, total_bytes);
        std::cout << "solvable_tableau_deals " << solvable << "\n";
        std::cout << "unsolvable_tableau_deals "
                  << total_deals - solvable << "\n";
        std::cout << "solvability_rate "
                  << static_cast<double>(solvable) / total_deals
                  << "\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        if (!options.deal.empty()) {
            return run_single_deal(options);
        }
        if (!options.deals_file.empty()) {
            return run_deals_file(options);
        }
        if (options.benchmark_deals > 0) {
            return run_benchmark(options);
        }
        if (options.stock_mode == "ordered") {
            return run_ordered_exhaustive(options);
        }
        return run_exhaustive(options);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
