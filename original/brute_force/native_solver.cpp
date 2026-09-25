#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int kSuits = 4;
constexpr int kMaxCards = 52;
constexpr int kMaxTableau = 7;
constexpr size_t kLegacyModelFeatureCount = 17;
constexpr size_t kPreviousModelFeatureCount = 31;
constexpr size_t kModelFeatureCount = 42;
constexpr uint8_t kFaceUp = 0x80;
constexpr uint8_t kCardMask = 0x7f;

struct Pile {
    std::array<uint8_t, kMaxCards> cards{};
    uint8_t size = 0;
};

struct State {
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

enum class MoveType {
    Draw,
    Recycle,
    WasteToFoundation,
    TableauToFoundation,
    WasteToTableau,
    TableauToTableau,
};

struct Candidate {
    int score;
    double model_score;
    State state;
};

int card_id(uint8_t card) { return card & kCardMask; }
bool face_up(uint8_t card) { return (card & kFaceUp) != 0; }
uint8_t make_face_up(uint8_t card) { return card | kFaceUp; }
uint8_t make_face_down(uint8_t card) { return card & kCardMask; }

class Solver {
public:
    Solver(
        int n,
        int tableau_count,
        bool model_enabled,
        bool model_only,
        int model_max_steps,
        const std::array<double, kModelFeatureCount>& model_weights,
        int max_recycles = 3,
        uint64_t exact_node_limit = 0,
        bool stock_as_reserve = false,
        int draw_count = 1,
        bool allow_tableau_stack_splitting = true
    )
        : n_(n),
          tableau_count_(tableau_count),
          model_enabled_(model_enabled),
          model_only_(model_only),
          model_max_steps_(model_max_steps),
          model_weights_(model_weights),
          max_recycles_(max_recycles),
          exact_node_limit_(exact_node_limit),
          stock_as_reserve_(stock_as_reserve),
          draw_count_(draw_count),
          allow_tableau_stack_splitting_(allow_tableau_stack_splitting) {
        if (stock_as_reserve_) {
            model_enabled_ = false;
        }
    }

    bool is_solvable(const State& initial) {
        if (model_enabled_) {
            ++model_attempts;
            if (model_solves(initial)) {
                ++model_wins;
                return true;
            }
            if (model_only_) {
                return false;
            }
        }
        ++exact_fallbacks;
        return exact_solvable(initial);
    }

    uint64_t expanded_positions = 0;
    uint64_t model_attempts = 0;
    uint64_t model_wins = 0;
    uint64_t model_steps = 0;
    uint64_t model_foundation_cards = 0;
    uint64_t model_foundation_cards_squared = 0;
    uint64_t model_cutoffs = 0;
    uint8_t last_model_foundation_cards = 0;
    uint64_t exact_fallbacks = 0;
    uint64_t exact_budget_exhaustions = 0;

private:
    int n_;
    int tableau_count_;
    bool model_enabled_;
    bool model_only_;
    int model_max_steps_;
    std::array<double, kModelFeatureCount> model_weights_;
    int max_recycles_;
    uint64_t exact_node_limit_;
    bool stock_as_reserve_;
    int draw_count_;
    bool allow_tableau_stack_splitting_;
    std::unordered_set<Key, KeyHash> known_unsolvable_;

    bool exact_solvable(const State& initial) {
        std::vector<State> pending;
        pending.push_back(initial);
        if (max_recycles_ < 0) {
            pending.back().recycles = 0;
        }
        std::unordered_set<Key, KeyHash> visited;
        visited.reserve(128);
        std::vector<Candidate> successors;
        successors.reserve(32);
        uint64_t local_expanded = 0;

        while (!pending.empty()) {
            State state = pending.back();
            pending.pop_back();
            Key key = position_key(state, true);
            if (known_unsolvable_.find(key) != known_unsolvable_.end()) {
                continue;
            }
            if (!visited.insert(key).second) {
                continue;
            }
            if (is_won(state)) {
                return true;
            }
            if (exact_node_limit_ > 0 && local_expanded >= exact_node_limit_) {
                ++exact_budget_exhaustions;
                return false;
            }

            ++expanded_positions;
            ++local_expanded;
            successors.clear();
            generate_successors(state, successors, true);
            std::sort(
                successors.begin(), successors.end(),
                [](const Candidate& left, const Candidate& right) {
                    return left.score < right.score;
                }
            );
            for (Candidate& successor : successors) {
                pending.push_back(std::move(successor.state));
            }
        }

        if (known_unsolvable_.size() + visited.size() > 1'000'000) {
            known_unsolvable_.clear();
        }
        known_unsolvable_.insert(visited.begin(), visited.end());
        return false;
    }

    void record_model_result(const State& state, bool cutoff) {
        uint64_t cards = std::accumulate(
            state.foundations.begin(), state.foundations.end(), uint64_t{0}
        );
        last_model_foundation_cards = static_cast<uint8_t>(cards);
        model_foundation_cards += cards;
        model_foundation_cards_squared += cards * cards;
        model_cutoffs += cutoff;
    }

    bool model_solves(const State& initial) {
        State state = initial;
        if (max_recycles_ < 0) {
            state.recycles = 0;
        }
        std::unordered_set<Key, KeyHash> seen;
        seen.reserve(512);
        seen.insert(position_key(state, false));
        std::vector<Candidate> moves;
        moves.reserve(32);

        int step = 0;
        for (; step < model_max_steps_ && !is_won(state); ++step) {
            moves.clear();
            generate_successors(state, moves, false);
            const Candidate* choice = nullptr;
            Key choice_key{};
            for (const Candidate& move : moves) {
                Key next_key = position_key(move.state, false);
                if (seen.find(next_key) != seen.end()) {
                    continue;
                }
                if (choice == nullptr || move.model_score > choice->model_score) {
                    choice = &move;
                    choice_key = next_key;
                }
            }
            if (choice == nullptr) {
                record_model_result(state, false);
                return false;
            }
            state = choice->state;
            seen.insert(choice_key);
            ++model_steps;
        }
        bool won = is_won(state);
        record_model_result(state, !won && step == model_max_steps_);
        return won;
    }

    int suit(uint8_t card) const { return card_id(card) / n_; }
    int rank(uint8_t card) const { return card_id(card) % n_ + 1; }
    int color(uint8_t card) const { return suit(card) % 2; }

    bool is_won(const State& state) const {
        return std::all_of(
            state.foundations.begin(), state.foundations.end(),
            [this](uint8_t foundation) { return foundation == n_; }
        );
    }

    bool can_foundation(const State& state, uint8_t card) const {
        return face_up(card) && state.foundations[suit(card)] + 1 == rank(card);
    }

    bool can_build(uint8_t card, const Pile& destination) const {
        if (!face_up(card)) {
            return false;
        }
        if (destination.size == 0) {
            return rank(card) == n_;
        }
        uint8_t top = destination.cards[destination.size - 1];
        return face_up(top) && rank(card) + 1 == rank(top) && color(card) != color(top);
    }

    bool safe_foundation(const State& state, uint8_t card) const {
        if (rank(card) == 1) {
            return true;
        }
        int minimum_opposite = n_;
        for (int foundation_suit = 0; foundation_suit < kSuits; ++foundation_suit) {
            if (foundation_suit % 2 != color(card)) {
                minimum_opposite = std::min(
                    minimum_opposite,
                    static_cast<int>(state.foundations[foundation_suit])
                );
            }
        }
        return minimum_opposite >= rank(card) - 1;
    }

    int hidden_cards(const Pile& pile) const {
        int hidden = 0;
        for (int i = 0; i < pile.size; ++i) {
            hidden += !face_up(pile.cards[i]);
        }
        return hidden;
    }

    bool packed_stack(const Pile& pile, int start) const {
        if (!allow_tableau_stack_splitting_ && start > 0 &&
            face_up(pile.cards[start - 1])) {
            return false;
        }
        for (int i = start; i < pile.size; ++i) {
            if (!face_up(pile.cards[i])) {
                return false;
            }
            if (i + 1 < pile.size &&
                (rank(pile.cards[i]) != rank(pile.cards[i + 1]) + 1 ||
                 color(pile.cards[i]) == color(pile.cards[i + 1]))) {
                return false;
            }
        }
        return true;
    }

    bool has_playable_king_move(const State& state) const {
        bool has_empty = false;
        for (int column = 0; column < tableau_count_; ++column) {
            has_empty = has_empty || state.tableau[column].size == 0;
        }
        if (!has_empty) {
            return false;
        }
        if (state.waste_size > 0 && rank(state.waste[state.waste_size - 1]) == n_) {
            return true;
        }
        for (int source = 0; source < tableau_count_; ++source) {
            const Pile& pile = state.tableau[source];
            for (int start = 0; start < pile.size; ++start) {
                if (rank(pile.cards[start]) == n_ && packed_stack(pile, start)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool exposed_card_playable(const State& state, uint8_t card) const {
        if (can_foundation(state, card)) {
            return true;
        }
        return count_tableau_destinations(state, card) > 0;
    }

    int count_tableau_destinations(const State& state, uint8_t card) const {
        int result = 0;
        for (int destination = 0; destination < tableau_count_; ++destination) {
            result += can_build(card, state.tableau[destination]);
        }
        return result;
    }

    int count_foundation_moves(const State& state) const {
        int result = state.waste_size > 0 &&
            can_foundation(state, state.waste[state.waste_size - 1]);
        for (int source = 0; source < tableau_count_; ++source) {
            const Pile& pile = state.tableau[source];
            result += pile.size > 0 && can_foundation(state, pile.cards[pile.size - 1]);
        }
        return result;
    }

    int count_reveal_moves(const State& state) const {
        int result = 0;
        for (int source = 0; source < tableau_count_; ++source) {
            const Pile& pile = state.tableau[source];
            if (pile.size < 2) {
                continue;
            }
            uint8_t top = pile.cards[pile.size - 1];
            if (!face_up(pile.cards[pile.size - 2]) && can_foundation(state, top)) {
                ++result;
            }
            for (int start = 1; start < pile.size; ++start) {
                if (face_up(pile.cards[start - 1]) || !packed_stack(pile, start)) {
                    continue;
                }
                for (int destination = 0; destination < tableau_count_; ++destination) {
                    if (destination != source &&
                        can_build(pile.cards[start], state.tableau[destination])) {
                        ++result;
                    }
                }
            }
        }
        return result;
    }

    int count_empty_source_moves(const State& state) const {
        int result = 0;
        for (int source = 0; source < tableau_count_; ++source) {
            const Pile& pile = state.tableau[source];
            if (pile.size == 0 || !face_up(pile.cards[pile.size - 1])) {
                continue;
            }
            if (pile.size == 1 && can_foundation(state, pile.cards[0])) {
                ++result;
            }
            if (!packed_stack(pile, 0)) {
                continue;
            }
            for (int destination = 0; destination < tableau_count_; ++destination) {
                if (destination != source &&
                    can_build(pile.cards[0], state.tableau[destination])) {
                    ++result;
                }
            }
        }
        return result;
    }

    int count_queen_moves_to_destination(
        const State& state,
        int destination
    ) const {
        if (destination < 0) {
            return 0;
        }
        int result = 0;
        if (state.waste_size > 0) {
            uint8_t card = state.waste[state.waste_size - 1];
            result += rank(card) == n_ - 1 &&
                can_build(card, state.tableau[destination]);
        }
        for (int source = 0; source < tableau_count_; ++source) {
            if (source == destination) {
                continue;
            }
            const Pile& pile = state.tableau[source];
            for (int start = 0; start < pile.size; ++start) {
                result += rank(pile.cards[start]) == n_ - 1 &&
                    packed_stack(pile, start) &&
                    can_build(pile.cards[start], state.tableau[destination]);
            }
        }
        return result;
    }

    int visible_foundation_support_demand(
        const State& state,
        uint8_t card
    ) const {
        if (rank(card) <= 1) {
            return 0;
        }
        int needed_rank = rank(card) - 1;
        int result = 0;
        if (state.waste_size > 0) {
            uint8_t candidate = state.waste[state.waste_size - 1];
            result += rank(candidate) == needed_rank &&
                color(candidate) != color(card);
        }
        for (int source = 0; source < tableau_count_; ++source) {
            const Pile& pile = state.tableau[source];
            for (int index = 0; index < pile.size; ++index) {
                uint8_t candidate = pile.cards[index];
                result += face_up(candidate) && rank(candidate) == needed_rank &&
                    color(candidate) != color(card);
            }
        }
        return result;
    }

    int foundation_safety_gap(const State& state, uint8_t card) const {
        int minimum_opposite = n_;
        for (int foundation_suit = 0; foundation_suit < kSuits; ++foundation_suit) {
            if (foundation_suit % 2 != color(card)) {
                minimum_opposite = std::min(
                    minimum_opposite,
                    static_cast<int>(state.foundations[foundation_suit])
                );
            }
        }
        return std::max(0, rank(card) - 1 - minimum_opposite);
    }

    double model_move_score(
        const State& state,
        const State& next,
        MoveType kind,
        int source,
        int destination,
        int count
    ) const {
        if (!model_enabled_) {
            return 0.0;
        }
        bool to_tableau = kind == MoveType::WasteToTableau ||
            kind == MoveType::TableauToTableau;
        bool foundation_move = kind == MoveType::WasteToFoundation ||
            kind == MoveType::TableauToFoundation;
        bool tableau_source = kind == MoveType::TableauToFoundation ||
            kind == MoveType::TableauToTableau;
        bool stock_action = kind == MoveType::Draw || kind == MoveType::Recycle;

        uint8_t foundation_card = 0;
        if (kind == MoveType::WasteToFoundation) {
            foundation_card = state.waste[state.waste_size - 1];
        } else if (kind == MoveType::TableauToFoundation) {
            const Pile& pile = state.tableau[source];
            foundation_card = pile.cards[pile.size - 1];
        }

        int source_hidden = tableau_source ? hidden_cards(state.tableau[source]) : 0;
        int remaining = tableau_source ? state.tableau[source].size - count : 0;
        bool reveals_hidden = tableau_source && remaining > 0 &&
            !face_up(state.tableau[source].cards[remaining - 1]);
        bool empties_source = tableau_source && count == state.tableau[source].size;

        bool fills_empty_with_king = false;
        if (to_tableau && state.tableau[destination].size == 0) {
            uint8_t moving_card = kind == MoveType::WasteToTableau
                ? state.waste[state.waste_size - 1]
                : state.tableau[source].cards[state.tableau[source].size - count];
            fills_empty_with_king = rank(moving_card) == n_;
        }

        int destination_hidden = 0;
        int destination_run = 0;
        bool blocks_foundation = false;
        if (to_tableau) {
            const Pile& pile = state.tableau[destination];
            destination_hidden = hidden_cards(pile);
            destination_run = pile.size - destination_hidden;
            blocks_foundation = pile.size > 0 &&
                can_foundation(state, pile.cards[pile.size - 1]);
        }

        bool all_columns_filled = true;
        for (int column = 0; column < tableau_count_; ++column) {
            all_columns_filled = all_columns_filled && state.tableau[column].size > 0;
        }
        bool creates_first_empty = all_columns_filled && empties_source;
        bool first_empty_with_king = creates_first_empty && has_playable_king_move(next);

        uint8_t revealed_card = 0;
        if (reveals_hidden) {
            const Pile& pile = next.tableau[source];
            revealed_card = pile.cards[pile.size - 1];
        }
        double revealed_card_low_rank = reveals_hidden
            ? static_cast<double>(n_ + 1 - rank(revealed_card))
            : 0.0;
        bool revealed_card_foundation_ready = reveals_hidden &&
            can_foundation(next, revealed_card);
        bool tableau_to_foundation = kind == MoveType::TableauToFoundation;
        bool tableau_to_tableau = kind == MoveType::TableauToTableau;
        bool non_reveal_tableau_move = tableau_to_tableau &&
            !reveals_hidden && !empties_source;
        double productive_stack_length = tableau_to_tableau &&
            (reveals_hidden || empties_source) ? static_cast<double>(count) : 0.0;
        bool draw_playable = kind == MoveType::Draw && next.waste_size > 0 &&
            exposed_card_playable(next, next.waste[next.waste_size - 1]);
        bool waste_unlocks_playable =
            (kind == MoveType::WasteToFoundation ||
             kind == MoveType::WasteToTableau) &&
            next.waste_size > 0 &&
            exposed_card_playable(next, next.waste[next.waste_size - 1]);
        int empty_king_queen_access = fills_empty_with_king
            ? count_queen_moves_to_destination(next, destination)
            : 0;
        int next_foundation_moves = count_foundation_moves(next);
        int next_reveal_moves = count_reveal_moves(next);
        int foundation_support_demand = foundation_move
            ? visible_foundation_support_demand(state, foundation_card)
            : 0;
        int recycle_pressure = kind == MoveType::Recycle && max_recycles_ >= 0
            ? state.recycles + 1 : 0;
        int draw_stock_remaining = kind == MoveType::Draw ? state.stock_size : 0;
        int revealed_card_tableau_moves = reveals_hidden
            ? count_tableau_destinations(next, revealed_card)
            : 0;
        int revealed_card_foundation_distance = reveals_hidden
            ? std::max(
                0,
                rank(revealed_card) -
                    (static_cast<int>(next.foundations[suit(revealed_card)]) + 1)
            )
            : 0;
        uint8_t drawn_card = kind == MoveType::Draw && next.waste_size > 0
            ? next.waste[next.waste_size - 1]
            : 0;
        bool draw_foundation_ready = kind == MoveType::Draw &&
            next.waste_size > 0 && can_foundation(next, drawn_card);
        int draw_tableau_moves = kind == MoveType::Draw && next.waste_size > 0
            ? count_tableau_destinations(next, drawn_card)
            : 0;
        bool waste_play = kind == MoveType::WasteToFoundation ||
            kind == MoveType::WasteToTableau;
        uint8_t unlocked_waste_card = waste_play && next.waste_size > 0
            ? next.waste[next.waste_size - 1]
            : 0;
        bool waste_unlocks_foundation_ready = waste_play && next.waste_size > 0 &&
            can_foundation(next, unlocked_waste_card);
        int waste_unlocks_tableau_moves = waste_play && next.waste_size > 0
            ? count_tableau_destinations(next, unlocked_waste_card)
            : 0;
        bool draw_buries_playable_waste = kind == MoveType::Draw &&
            state.waste_size > 0 &&
            exposed_card_playable(state, state.waste[state.waste_size - 1]);
        int waste_play_recycle_pressure = waste_play && max_recycles_ >= 0
            ? state.recycles : 0;
        int next_empty_source_moves = count_empty_source_moves(next);
        int foundation_lag = foundation_move
            ? static_cast<int>(*std::max_element(
                state.foundations.begin(), state.foundations.end()
            )) - static_cast<int>(state.foundations[suit(foundation_card)])
            : 0;
        int unsafe_foundation_distance = foundation_move
            ? foundation_safety_gap(state, foundation_card)
            : 0;

        std::array<double, kModelFeatureCount> features{
            foundation_move && safe_foundation(state, foundation_card) ? 1.0 : 0.0,
            reveals_hidden ? 1.0 : 0.0,
            fills_empty_with_king ? 1.0 : 0.0,
            to_tableau ? static_cast<double>(count) : 0.0,
            stock_action ? 1.0 : 0.0,
            foundation_move ? 1.0 : 0.0,
            reveals_hidden ? static_cast<double>(source_hidden) : 0.0,
            empties_source ? 1.0 : 0.0,
            kind == MoveType::WasteToTableau ? 1.0 : 0.0,
            kind == MoveType::Recycle ? 1.0 : 0.0,
            static_cast<double>(destination_hidden),
            static_cast<double>(destination_run),
            blocks_foundation ? 1.0 : 0.0,
            foundation_move ? static_cast<double>(rank(foundation_card)) : 0.0,
            static_cast<double>(source_hidden),
            creates_first_empty ? 1.0 : 0.0,
            first_empty_with_king ? 1.0 : 0.0,
            revealed_card_low_rank,
            revealed_card_foundation_ready ? 1.0 : 0.0,
            tableau_to_foundation ? 1.0 : 0.0,
            tableau_to_tableau ? 1.0 : 0.0,
            non_reveal_tableau_move ? 1.0 : 0.0,
            productive_stack_length,
            draw_playable ? 1.0 : 0.0,
            waste_unlocks_playable ? 1.0 : 0.0,
            static_cast<double>(empty_king_queen_access),
            static_cast<double>(next_foundation_moves),
            static_cast<double>(next_reveal_moves),
            static_cast<double>(foundation_support_demand),
            static_cast<double>(recycle_pressure),
            static_cast<double>(draw_stock_remaining),
            static_cast<double>(revealed_card_tableau_moves),
            static_cast<double>(revealed_card_foundation_distance),
            draw_foundation_ready ? 1.0 : 0.0,
            static_cast<double>(draw_tableau_moves),
            waste_unlocks_foundation_ready ? 1.0 : 0.0,
            static_cast<double>(waste_unlocks_tableau_moves),
            draw_buries_playable_waste ? 1.0 : 0.0,
            static_cast<double>(waste_play_recycle_pressure),
            static_cast<double>(next_empty_source_moves),
            static_cast<double>(foundation_lag),
            static_cast<double>(unsafe_foundation_distance),
        };
        double score = 0.0;
        for (size_t index = 0; index < features.size(); ++index) {
            score += model_weights_[index] * features[index];
        }
        return score;
    }

    void reveal_top(Pile& pile) const {
        if (pile.size > 0) {
            pile.cards[pile.size - 1] = make_face_up(pile.cards[pile.size - 1]);
        }
    }

    Key position_key(const State& state, bool canonical_tableau) const {
        std::array<int, kMaxTableau> order{};
        std::iota(order.begin(), order.begin() + tableau_count_, 0);
        if (canonical_tableau) {
            std::sort(
                order.begin(), order.begin() + tableau_count_,
                [&state](int left, int right) {
                    const Pile& a = state.tableau[left];
                    const Pile& b = state.tableau[right];
                    int common = std::min(a.size, b.size);
                    for (int i = 0; i < common; ++i) {
                        if (a.cards[i] != b.cards[i]) {
                            return a.cards[i] < b.cards[i];
                        }
                    }
                    return a.size < b.size;
                }
            );
        }

        Key key{};
        size_t cursor = 0;
        for (int position = 0; position < tableau_count_; ++position) {
            const Pile& pile = state.tableau[order[position]];
            key.bytes[cursor++] = pile.size;
            for (int i = 0; i < pile.size; ++i) {
                key.bytes[cursor++] = pile.cards[i];
            }
        }
        key.bytes[cursor++] = state.stock_size;
        if (stock_as_reserve_) {
            std::array<uint8_t, kMaxCards> sorted_stock{};
            std::copy(
                state.stock.begin(),
                state.stock.begin() + state.stock_size,
                sorted_stock.begin()
            );
            std::sort(sorted_stock.begin(), sorted_stock.begin() + state.stock_size);
            for (int i = 0; i < state.stock_size; ++i) {
                key.bytes[cursor++] = sorted_stock[i];
            }
        } else {
            for (int i = 0; i < state.stock_size; ++i) {
                key.bytes[cursor++] = state.stock[i];
            }
        }
        key.bytes[cursor++] = state.waste_size;
        for (int i = 0; i < state.waste_size; ++i) {
            key.bytes[cursor++] = state.waste[i];
        }
        for (uint8_t foundation : state.foundations) {
            key.bytes[cursor++] = foundation;
        }
        // Unlimited passes have no remaining-pass resource. Including the
        // historical count would turn stock cycles into distinct positions.
        key.bytes[cursor++] = max_recycles_ < 0 ? 0 : state.recycles;
        return key;
    }

    void generate_successors(
        const State& state,
        std::vector<Candidate>& result,
        bool force_safe_foundation
    ) const {
        bool has_forced_foundation = false;
        if (force_safe_foundation && stock_as_reserve_) {
            for (int stock_index = 0; stock_index < state.stock_size; ++stock_index) {
                uint8_t card = make_face_up(state.stock[stock_index]);
                if (!can_foundation(state, card) || !safe_foundation(state, card)) {
                    continue;
                }
                State next = state;
                for (int i = stock_index; i + 1 < next.stock_size; ++i) {
                    next.stock[i] = next.stock[i + 1];
                }
                --next.stock_size;
                ++next.foundations[suit(card)];
                result.push_back({200, 0.0, std::move(next)});
                has_forced_foundation = true;
            }
        }
        if (force_safe_foundation && state.waste_size > 0) {
            uint8_t card = state.waste[state.waste_size - 1];
            if (can_foundation(state, card) && safe_foundation(state, card)) {
                State next = state;
                --next.waste_size;
                ++next.foundations[suit(card)];
                double model_score = model_move_score(
                    state, next, MoveType::WasteToFoundation, -1, -1, 1
                );
                result.push_back({200, model_score, std::move(next)});
                has_forced_foundation = true;
            }
        }
        for (int source = 0; force_safe_foundation && source < tableau_count_; ++source) {
            const Pile& pile = state.tableau[source];
            if (pile.size == 0) {
                continue;
            }
            uint8_t card = pile.cards[pile.size - 1];
            if (!can_foundation(state, card) || !safe_foundation(state, card)) {
                continue;
            }
            State next = state;
            --next.tableau[source].size;
            ++next.foundations[suit(card)];
            bool revealed = next.tableau[source].size > 0 &&
                !face_up(next.tableau[source].cards[next.tableau[source].size - 1]);
            reveal_top(next.tableau[source]);
            double model_score = model_move_score(
                state, next, MoveType::TableauToFoundation, source, -1, 1
            );
            result.push_back({revealed ? 300 : 200, model_score, std::move(next)});
            has_forced_foundation = true;
        }
        if (has_forced_foundation) {
            return;
        }

        if (stock_as_reserve_) {
            for (int stock_index = 0; stock_index < state.stock_size; ++stock_index) {
                uint8_t card = make_face_up(state.stock[stock_index]);
                if (can_foundation(state, card)) {
                    State next = state;
                    for (int i = stock_index; i + 1 < next.stock_size; ++i) {
                        next.stock[i] = next.stock[i + 1];
                    }
                    --next.stock_size;
                    ++next.foundations[suit(card)];
                    result.push_back({200, 0.0, std::move(next)});
                }
                bool used_empty_destination = false;
                for (int destination = 0; destination < tableau_count_; ++destination) {
                    if (!can_build(card, state.tableau[destination])) {
                        continue;
                    }
                    if (state.tableau[destination].size == 0) {
                        if (used_empty_destination) {
                            continue;
                        }
                        used_empty_destination = true;
                    }
                    State next = state;
                    for (int i = stock_index; i + 1 < next.stock_size; ++i) {
                        next.stock[i] = next.stock[i + 1];
                    }
                    --next.stock_size;
                    next.tableau[destination]
                        .cards[next.tableau[destination].size++] = card;
                    result.push_back({40, 0.0, std::move(next)});
                }
            }
        }

        if (!stock_as_reserve_ && state.stock_size > 0) {
            State next = state;
            int cards_to_draw = std::min(draw_count_, static_cast<int>(next.stock_size));
            for (int index = 0; index < cards_to_draw; ++index) {
                uint8_t card = next.stock[--next.stock_size];
                next.waste[next.waste_size++] = make_face_up(card);
            }
            double model_score = model_move_score(
                state, next, MoveType::Draw, -1, -1, 1
            );
            result.push_back({10, model_score, std::move(next)});
        } else if (!stock_as_reserve_ && state.waste_size > 0 &&
                   (max_recycles_ < 0 || state.recycles < max_recycles_)) {
            State next = state;
            next.stock_size = next.waste_size;
            for (int i = 0; i < next.waste_size; ++i) {
                next.stock[i] = make_face_down(next.waste[next.waste_size - 1 - i]);
            }
            next.waste_size = 0;
            next.recycles = max_recycles_ < 0 ? 0 : next.recycles + 1;
            double model_score = model_move_score(
                state, next, MoveType::Recycle, -1, -1, 1
            );
            result.push_back({0, model_score, std::move(next)});
        }

        if (!stock_as_reserve_ && state.waste_size > 0) {
            uint8_t card = state.waste[state.waste_size - 1];
            if (can_foundation(state, card)) {
                State next = state;
                --next.waste_size;
                ++next.foundations[suit(card)];
                double model_score = model_move_score(
                    state, next, MoveType::WasteToFoundation, -1, -1, 1
                );
                result.push_back({200, model_score, std::move(next)});
            }
            bool used_empty_destination = false;
            for (int destination = 0; destination < tableau_count_; ++destination) {
                if (!can_build(card, state.tableau[destination])) {
                    continue;
                }
                if (state.tableau[destination].size == 0) {
                    if (used_empty_destination) {
                        continue;
                    }
                    used_empty_destination = true;
                }
                State next = state;
                --next.waste_size;
                next.tableau[destination].cards[next.tableau[destination].size++] = card;
                double model_score = model_move_score(
                    state, next, MoveType::WasteToTableau, -1, destination, 1
                );
                result.push_back({40, model_score, std::move(next)});
            }
        }

        for (int source = 0; source < tableau_count_; ++source) {
            const Pile& source_pile = state.tableau[source];
            if (source_pile.size == 0 || !face_up(source_pile.cards[source_pile.size - 1])) {
                continue;
            }

            uint8_t top = source_pile.cards[source_pile.size - 1];
            if (can_foundation(state, top)) {
                State next = state;
                --next.tableau[source].size;
                ++next.foundations[suit(top)];
                bool revealed = next.tableau[source].size > 0 &&
                    !face_up(next.tableau[source].cards[next.tableau[source].size - 1]);
                reveal_top(next.tableau[source]);
                double model_score = model_move_score(
                    state, next, MoveType::TableauToFoundation, source, -1, 1
                );
                result.push_back({revealed ? 300 : 200, model_score, std::move(next)});
            }

            for (int start = 0; start < source_pile.size; ++start) {
                if (!packed_stack(source_pile, start)) {
                    continue;
                }

                uint8_t moving_card = source_pile.cards[start];
                bool used_empty_destination = false;
                for (int destination = 0; destination < tableau_count_; ++destination) {
                    if (destination == source ||
                        !can_build(moving_card, state.tableau[destination])) {
                        continue;
                    }
                    if (state.tableau[destination].size == 0) {
                        if (start == 0) {
                            continue;
                        }
                        if (used_empty_destination) {
                            continue;
                        }
                        used_empty_destination = true;
                    }
                    State next = state;
                    Pile& next_source = next.tableau[source];
                    Pile& next_destination = next.tableau[destination];
                    for (int i = start; i < source_pile.size; ++i) {
                        next_destination.cards[next_destination.size++] = source_pile.cards[i];
                    }
                    next_source.size = start;
                    bool revealed = next_source.size > 0 &&
                        !face_up(next_source.cards[next_source.size - 1]);
                    reveal_top(next_source);
                    int moved_cards = source_pile.size - start;
                    double model_score = model_move_score(
                        state,
                        next,
                        MoveType::TableauToTableau,
                        source,
                        destination,
                        moved_cards
                    );
                    result.push_back(
                        {revealed ? 250 : 50, model_score, std::move(next)}
                    );
                }
            }
        }
    }
};

struct Options {
    int n = 3;
    int threads = 1;
    uint64_t max_patterns = 0;
    uint64_t benchmark_deals = 0;
    uint64_t benchmark_seed = 20260714;
    uint64_t exact_node_limit = 0;
    uint64_t stock_order_tableaus = 0;
    bool model_enabled = true;
    bool model_only = false;
    bool allow_huge_run = false;
    bool collapse_stock_order = false;
    bool stock_as_reserve = false;
    int model_max_steps = 300;
    int max_recycles = 3;
    int draw_count = 1;
    bool allow_tableau_stack_splitting = true;
    std::array<double, kModelFeatureCount> model_weights{
        8.042521340737478,
        5.079509858778847,
        1.0715637483026381,
        -1.0315555151035165,
        -1.0301777716066625,
        -0.07867712926652164,
        1.0502653673004092,
        1.0122255109713585,
        2.2163598526425887,
        -2.043119906477297,
        -0.06887841479579193,
        0.04678845358001803,
        0.137402118160686,
        -0.10838485806665588,
        0.6370608072236218,
        1.0280564066869964,
        2.0187127117822232,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    };
    std::string output;
    std::string completion;
    std::string verify_stock_order;
    std::string benchmark_outcomes;
    std::string benchmark_foundations;
};

uint64_t factorial(int value) {
    uint64_t result = 1;
    for (int i = 2; i <= value; ++i) {
        result *= i;
    }
    return result;
}

int default_tableau_count(int n) {
    int target = 2 * n;
    int columns = 1;
    while (columns * (columns + 1) / 2 < target) {
        ++columns;
    }
    return columns;
}

void generate_patterns_recursive(
    int n,
    std::array<int, kSuits>& counts,
    std::vector<uint8_t>& pattern,
    bool seen_suit_one,
    std::vector<std::vector<uint8_t>>& output
) {
    if (static_cast<int>(pattern.size()) == 4 * n) {
        output.push_back(pattern);
        return;
    }
    for (int suit = 0; suit < kSuits; ++suit) {
        if (counts[suit] == 0 || (suit == 3 && !seen_suit_one)) {
            continue;
        }
        --counts[suit];
        pattern.push_back(static_cast<uint8_t>(suit));
        generate_patterns_recursive(
            n, counts, pattern, seen_suit_one || suit == 1, output
        );
        pattern.pop_back();
        ++counts[suit];
    }
}

std::vector<std::vector<uint8_t>> generate_patterns(int n) {
    std::vector<std::vector<uint8_t>> patterns;
    std::array<int, kSuits> counts{n - 1, n, n, n};
    std::vector<uint8_t> pattern{0};
    generate_patterns_recursive(n, counts, pattern, false, patterns);
    return patterns;
}

std::vector<std::vector<uint8_t>> rank_permutations(int n) {
    std::vector<uint8_t> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<std::vector<uint8_t>> output;
    do {
        output.push_back(permutation);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return output;
}

State initial_state(const std::vector<uint8_t>& deal, int tableau_count) {
    State state{};
    int cursor = 0;
    for (int column = 0; column < tableau_count; ++column) {
        Pile& pile = state.tableau[column];
        for (int row = 0; row <= column && cursor < static_cast<int>(deal.size()); ++row) {
            uint8_t card = deal[cursor++];
            pile.cards[pile.size++] = row == column ? make_face_up(card) : card;
        }
        if (pile.size > 0) {
            pile.cards[pile.size - 1] = make_face_up(pile.cards[pile.size - 1]);
        }
    }
    for (int index = static_cast<int>(deal.size()) - 1; index >= cursor; --index) {
        state.stock[state.stock_size++] = deal[index];
    }
    return state;
}

bool has_static_blocked_card(
    const std::vector<uint8_t>& deal,
    int n,
    int tableau_count
) {
    int pile_start = 0;
    for (int column = 0; column < tableau_count; ++column) {
        int pile_size = column + 1;
        for (int row = 3; row < pile_size; ++row) {
            int target = deal[pile_start + row];
            int target_suit = target / n;
            int target_rank = target % n + 1;
            if (target_rank <= 1 || target_rank >= n) {
                continue;
            }

            bool suit_blocked = false;
            std::array<bool, 2> support_blocked{};
            for (int lower_row = 0; lower_row < row; ++lower_row) {
                int lower = deal[pile_start + lower_row];
                int lower_suit = lower / n;
                int lower_rank = lower % n + 1;
                if (lower_suit == target_suit && lower_rank < target_rank) {
                    suit_blocked = true;
                }
                if (lower_rank != target_rank + 1 ||
                    lower_suit % 2 == target_suit % 2) {
                    continue;
                }
                support_blocked[lower_suit / 2] = true;
            }
            if (suit_blocked && support_blocked[0] && support_blocked[1]) {
                return true;
            }
        }
        pile_start += pile_size;
    }
    return false;
}

void assign_ranks_to_pattern(
    const std::vector<uint8_t>& pattern,
    const std::vector<std::vector<uint8_t>>& ranks,
    uint64_t combination,
    int n,
    std::vector<uint8_t>& deal
) {
    uint64_t remainder = combination;
    std::array<size_t, kSuits> rank_order{};
    for (int suit = kSuits - 1; suit >= 0; --suit) {
        rank_order[suit] = remainder % ranks.size();
        remainder /= ranks.size();
    }
    std::array<size_t, kSuits> occurrence{};
    for (size_t position = 0; position < pattern.size(); ++position) {
        int suit = pattern[position];
        uint8_t rank_index = ranks[rank_order[suit]][occurrence[suit]++];
        deal[position] = static_cast<uint8_t>(suit * n + rank_index);
    }
}

uint64_t canonical_tableau_key(
    const std::vector<uint8_t>& deal,
    int n,
    int tableau_cards
) {
    uint64_t best = UINT64_MAX;
    for (int swap_colors = 0; swap_colors < 2; ++swap_colors) {
        for (int swap_color_zero = 0; swap_color_zero < 2; ++swap_color_zero) {
            for (int swap_color_one = 0; swap_color_one < 2; ++swap_color_one) {
                uint64_t key = 0;
                for (int position = 0; position < tableau_cards; ++position) {
                    int card = deal[position];
                    int source_suit = card / n;
                    int rank_index = card % n;
                    int color = source_suit % 2;
                    int suit_in_color = source_suit / 2;
                    int swap_within = color == 0 ? swap_color_zero : swap_color_one;
                    int mapped_color = color ^ swap_colors;
                    int mapped_in_color = suit_in_color ^ swap_within;
                    int mapped_suit = 2 * mapped_in_color + mapped_color;
                    int mapped_card = mapped_suit * n + rank_index;
                    key = (key << 4) | static_cast<uint64_t>(mapped_card);
                }
                best = std::min(best, key);
            }
        }
    }
    return best;
}

bool read_exact(int file, void* buffer, size_t size, off_t offset) {
    uint8_t* destination = static_cast<uint8_t*>(buffer);
    size_t total = 0;
    while (total < size) {
        ssize_t count = pread(file, destination + total, size - total, offset + total);
        if (count <= 0) {
            return false;
        }
        total += static_cast<size_t>(count);
    }
    return true;
}

bool write_exact(int file, const void* buffer, size_t size, off_t offset) {
    const uint8_t* source = static_cast<const uint8_t*>(buffer);
    size_t total = 0;
    while (total < size) {
        ssize_t count = pwrite(file, source + total, size - total, offset + total);
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

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string argument = argv[i];
        if (argument == "--disable-model") {
            options.model_enabled = false;
            continue;
        }
        if (argument == "--model-only") {
            options.model_enabled = true;
            options.model_only = true;
            continue;
        }
        if (argument == "--allow-huge-run") {
            options.allow_huge_run = true;
            continue;
        }
        if (argument == "--collapse-stock-order") {
            options.collapse_stock_order = true;
            continue;
        }
        if (argument == "--stock-as-reserve") {
            options.stock_as_reserve = true;
            continue;
        }
        if (argument == "--no-tableau-splitting") {
            options.allow_tableau_stack_splitting = false;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        std::string value = argv[++i];
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
        } else if (argument == "--benchmark-outcomes") {
            if (value.empty()) {
                throw std::runtime_error("--benchmark-outcomes requires a path");
            }
            options.benchmark_outcomes = value;
        } else if (argument == "--benchmark-foundations") {
            if (value.empty()) {
                throw std::runtime_error("--benchmark-foundations requires a path");
            }
            options.benchmark_foundations = value;
        } else if (argument == "--exact-node-limit") {
            options.exact_node_limit = std::stoull(value);
        } else if (argument == "--check-stock-orders") {
            options.stock_order_tableaus = std::stoull(value);
        } else if (argument == "--verify-stock-order") {
            options.verify_stock_order = value;
        } else if (argument == "--model-max-steps") {
            options.model_max_steps = std::stoi(value);
        } else if (argument == "--max-recycles") {
            options.max_recycles = std::stoi(value);
        } else if (argument == "--draw-count") {
            options.draw_count = std::stoi(value);
        } else if (argument == "--model-weights") {
            std::stringstream stream(value);
            std::string item;
            size_t index = 0;
            while (std::getline(stream, item, ',')) {
                if (index >= options.model_weights.size()) {
                    throw std::runtime_error("too many model weights");
                }
                options.model_weights[index++] = std::stod(item);
            }
            if (index != kLegacyModelFeatureCount &&
                index != kPreviousModelFeatureCount &&
                index != options.model_weights.size()) {
                throw std::runtime_error("expected 17, 31, or 42 model weights");
            }
        } else if (argument == "--output") {
            options.output = value;
        } else if (argument == "--completion") {
            options.completion = value;
        } else {
            throw std::runtime_error("unknown argument " + argument);
        }
    }
    if (options.draw_count != 1 && options.draw_count != 3) {
        throw std::runtime_error("--draw-count must be 1 or 3");
    }
    if (options.max_recycles < -1 || options.max_recycles > UINT8_MAX) {
        throw std::runtime_error("--max-recycles must be -1 (unlimited) or 0 through 255");
    }
    bool variant = options.draw_count != 1 || options.max_recycles != 3 ||
        !options.allow_tableau_stack_splitting;
    if (variant &&
        (options.benchmark_deals == 0 || !options.model_only ||
         !options.model_enabled || options.stock_order_tableaus != 0 ||
         !options.verify_stock_order.empty() || options.collapse_stock_order ||
         options.stock_as_reserve)) {
        throw std::runtime_error(
            "rule variants require a model-only benchmark without stock-order/reserve modes"
        );
    }
    if (!options.benchmark_foundations.empty() &&
        (options.benchmark_deals == 0 || !options.model_enabled ||
         !options.model_only || options.stock_order_tableaus != 0 ||
         !options.verify_stock_order.empty() || options.collapse_stock_order ||
         options.stock_as_reserve)) {
        throw std::runtime_error(
            "--benchmark-foundations requires a model-only benchmark without stock-order/reserve modes"
        );
    }
    if (!options.benchmark_foundations.empty() &&
        options.benchmark_foundations == options.benchmark_outcomes) {
        throw std::runtime_error("benchmark foundations and outcomes need different paths");
    }
    if (!options.benchmark_outcomes.empty() &&
        (options.benchmark_deals == 0 || options.stock_order_tableaus != 0 ||
         !options.verify_stock_order.empty() || options.collapse_stock_order)) {
        throw std::runtime_error(
            "--benchmark-outcomes requires a benchmark without stock-order modes"
        );
    }
    if (options.n < 2 || options.n > 13 || options.threads < 1 ||
        options.model_max_steps < 1 ||
        (options.benchmark_deals == 0 && options.stock_order_tableaus == 0 &&
         options.verify_stock_order.empty() &&
         (options.output.empty() || options.completion.empty()))) {
        throw std::runtime_error(
            "require 2 <= n <= 4, threads >= 1, and checkpoint paths unless benchmarking"
        );
    }
    if (options.collapse_stock_order && options.n != 4) {
        throw std::runtime_error("--collapse-stock-order currently requires n=4");
    }
    if (options.n > 4 &&
        (options.benchmark_deals == 0 ||
         (!options.model_only && options.exact_node_limit == 0) ||
         options.stock_order_tableaus != 0 || !options.verify_stock_order.empty())) {
        throw std::runtime_error(
            "n > 4 requires a model-only benchmark or a positive exact node limit"
        );
    }
    if (options.n == 4 && !options.collapse_stock_order &&
        options.benchmark_deals == 0 &&
        options.stock_order_tableaus == 0 && options.verify_stock_order.empty() &&
        !options.allow_huge_run) {
        throw std::runtime_error(
            "n=4 needs about 327 GB for its raw bitset; pass --allow-huge-run explicitly"
        );
    }
    return options;
}

uint64_t popcount_bytes(const std::vector<uint8_t>& bytes) {
    uint64_t total = 0;
    for (uint8_t byte : bytes) {
        total += static_cast<uint64_t>(__builtin_popcount(byte));
    }
    return total;
}

int run_benchmark(const Options& options) {
    int tableau_count = default_tableau_count(options.n);
    // Each worker owns one distinct byte per deterministic deal index.
    // A zero means no accepted win; bounded/model-only searches may be unresolved.
    std::vector<uint8_t> outcomes;
    if (!options.benchmark_outcomes.empty()) {
        outcomes.resize(options.benchmark_deals);
    }
    std::vector<uint8_t> foundations;
    if (!options.benchmark_foundations.empty()) {
        foundations.resize(options.benchmark_deals);
    }
    std::atomic<uint64_t> next_deal{0};
    std::atomic<uint64_t> solvable{0};
    std::atomic<uint64_t> model_attempts{0};
    std::atomic<uint64_t> model_wins{0};
    std::atomic<uint64_t> model_steps{0};
    std::atomic<uint64_t> model_foundation_cards{0};
    std::atomic<uint64_t> model_foundation_cards_squared{0};
    std::atomic<uint64_t> model_cutoffs{0};
    std::atomic<uint64_t> exact_fallbacks{0};
    std::atomic<uint64_t> exact_budget_exhaustions{0};
    std::atomic<uint64_t> expanded{0};
    std::atomic<uint64_t> static_blocked{0};
    auto started = std::chrono::steady_clock::now();

    auto worker = [&]() {
        Solver solver(
            options.n,
            tableau_count,
            options.model_enabled,
            options.model_only,
            options.model_max_steps,
            options.model_weights,
            options.max_recycles,
            options.exact_node_limit,
            false,
            options.draw_count,
            options.allow_tableau_stack_splitting
        );
        std::vector<uint8_t> deal(4 * options.n);
        while (true) {
            uint64_t index = next_deal.fetch_add(1);
            if (index >= options.benchmark_deals) {
                break;
            }
            std::iota(deal.begin(), deal.end(), 0);
            std::mt19937_64 random(
                options.benchmark_seed + index * 0x9e3779b97f4a7c15ULL
            );
            std::shuffle(deal.begin(), deal.end(), random);
            static_blocked.fetch_add(
                has_static_blocked_card(deal, options.n, tableau_count) ? 1 : 0
            );
            uint8_t outcome = solver.is_solvable(
                initial_state(deal, tableau_count)
            ) ? 1 : 0;
            solvable.fetch_add(outcome);
            if (!outcomes.empty()) {
                outcomes[index] = outcome;
            }
            if (!foundations.empty()) {
                foundations[index] = solver.last_model_foundation_cards;
            }
        }
        model_attempts.fetch_add(solver.model_attempts);
        model_wins.fetch_add(solver.model_wins);
        model_steps.fetch_add(solver.model_steps);
        model_foundation_cards.fetch_add(solver.model_foundation_cards);
        model_foundation_cards_squared.fetch_add(solver.model_foundation_cards_squared);
        model_cutoffs.fetch_add(solver.model_cutoffs);
        exact_fallbacks.fetch_add(solver.exact_fallbacks);
        exact_budget_exhaustions.fetch_add(solver.exact_budget_exhaustions);
        expanded.fetch_add(solver.expanded_positions);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < options.threads; ++i) {
        threads.emplace_back(worker);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    if (!options.benchmark_outcomes.empty()) {
        int file = open(
            options.benchmark_outcomes.c_str(),
            O_WRONLY | O_CREAT | O_TRUNC,
            0644
        );
        if (file < 0) {
            throw std::runtime_error("could not open benchmark outcomes file");
        }
        bool written = write_exact(file, outcomes.data(), outcomes.size(), 0);
        int close_result = close(file);
        if (!written || close_result != 0) {
            throw std::runtime_error("could not write benchmark outcomes file");
        }
        std::cout << "benchmark_outcomes " << options.benchmark_outcomes << "\n";
        std::cout << "benchmark_outcomes_bytes " << outcomes.size() << "\n";
    }
    if (!options.benchmark_foundations.empty()) {
        int file = open(
            options.benchmark_foundations.c_str(),
            O_WRONLY | O_CREAT | O_TRUNC,
            0644
        );
        if (file < 0) {
            throw std::runtime_error("could not open benchmark foundations file");
        }
        bool written = write_exact(file, foundations.data(), foundations.size(), 0);
        int close_result = close(file);
        if (!written || close_result != 0) {
            throw std::runtime_error("could not write benchmark foundations file");
        }
        std::cout << "benchmark_foundations " << options.benchmark_foundations << "\n";
        std::cout << "benchmark_foundations_bytes " << foundations.size() << "\n";
    }
    std::cout << "benchmark_deals " << options.benchmark_deals << "\n";
    std::cout << "draw_count " << options.draw_count << "\n";
    std::cout << "max_recycles " << options.max_recycles << "\n";
    std::cout << "allow_tableau_stack_splitting "
              << options.allow_tableau_stack_splitting << "\n";
    std::cout << "solvable_deals " << solvable.load() << "\n";
    std::cout << "model_attempts " << model_attempts.load() << "\n";
    std::cout << "model_wins " << model_wins.load() << "\n";
    std::cout << "model_steps " << model_steps.load() << "\n";
    std::cout << "model_foundation_cards " << model_foundation_cards.load() << "\n";
    std::cout << "model_foundation_cards_squared "
              << model_foundation_cards_squared.load() << "\n";
    std::cout << "model_cutoffs " << model_cutoffs.load() << "\n";
    std::cout << "exact_fallbacks " << exact_fallbacks.load() << "\n";
    std::cout << "exact_budget_exhaustions "
              << exact_budget_exhaustions.load() << "\n";
    std::cout << "expanded_positions " << expanded.load() << "\n";
    std::cout << "static_blocked_deals " << static_blocked.load() << "\n";
    std::cout << "elapsed_seconds " << elapsed << "\n";
    std::cout << "deals_per_second " << options.benchmark_deals / elapsed << "\n";
    return 0;
}

int verify_stock_order_invariance(const Options& options) {
    auto patterns = generate_patterns(options.n);
    auto ranks = rank_permutations(options.n);
    uint64_t deals_per_pattern = 1;
    for (int suit = 0; suit < kSuits; ++suit) {
        deals_per_pattern *= ranks.size();
    }
    uint64_t total_deals = patterns.size() * deals_per_pattern;
    size_t expected_bytes = static_cast<size_t>((total_deals + 7) / 8);
    int file = open(options.verify_stock_order.c_str(), O_RDONLY);
    if (file < 0) {
        throw std::runtime_error("could not open saved bitset");
    }
    struct stat info{};
    if (fstat(file, &info) != 0 || static_cast<size_t>(info.st_size) != expected_bytes) {
        close(file);
        throw std::runtime_error("saved bitset has the wrong size");
    }
    std::vector<uint8_t> bits(expected_bytes);
    if (!read_exact(file, bits.data(), bits.size(), 0)) {
        close(file);
        throw std::runtime_error("could not read saved bitset");
    }
    close(file);

    int tableau_count = default_tableau_count(options.n);
    int tableau_cards = tableau_count * (tableau_count + 1) / 2;
    std::unordered_map<uint64_t, uint8_t> outcomes;
    outcomes.reserve(1'000'000);
    std::vector<uint8_t> deal(4 * options.n);
    uint64_t deal_index = 0;
    auto started = std::chrono::steady_clock::now();

    for (size_t pattern_index = 0; pattern_index < patterns.size(); ++pattern_index) {
        for (uint64_t combination = 0; combination < deals_per_pattern; ++combination) {
            assign_ranks_to_pattern(
                patterns[pattern_index], ranks, combination, options.n, deal
            );
            bool solvable = (bits[deal_index / 8] & (1U << (deal_index % 8))) != 0;
            uint64_t key = canonical_tableau_key(deal, options.n, tableau_cards);
            uint8_t& status = outcomes[key];
            status |= solvable ? 1 : 2;
            if (status == 3) {
                std::cout << "stock_order_invariant false\n";
                std::cout << "mixed_tableau_key " << key << "\n";
                std::cout << "deal_index " << deal_index << "\n";
                return 1;
            }
            ++deal_index;
        }
        if ((pattern_index + 1) % 5000 == 0) {
            std::cout << "verified_patterns " << pattern_index + 1 << "/"
                      << patterns.size() << "\n" << std::flush;
        }
    }

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    std::cout << "stock_order_invariant true\n";
    std::cout << "canonical_deals_checked " << deal_index << "\n";
    std::cout << "canonical_tableau_groups " << outcomes.size() << "\n";
    std::cout << "elapsed_seconds " << elapsed << "\n";
    return 0;
}

int check_stock_orders(const Options& options) {
    int tableau_count = default_tableau_count(options.n);
    int tableau_cards = tableau_count * (tableau_count + 1) / 2;
    std::atomic<uint64_t> next_tableau{0};
    std::atomic<uint64_t> checked_tableaus{0};
    std::atomic<uint64_t> checked_deals{0};
    std::atomic<bool> mismatch{false};
    std::mutex mismatch_mutex;
    std::string mismatch_description;
    auto started = std::chrono::steady_clock::now();

    auto worker = [&]() {
        Solver solver(
            options.n,
            tableau_count,
            false,
            false,
            options.model_max_steps,
            options.model_weights,
            options.max_recycles,
            options.exact_node_limit,
            false,
            options.draw_count,
            options.allow_tableau_stack_splitting
        );
        std::vector<uint8_t> shuffled(4 * options.n);
        std::vector<uint8_t> deal(4 * options.n);
        std::vector<uint8_t> stock(4 * options.n - tableau_cards);
        while (!mismatch.load()) {
            uint64_t tableau_index = next_tableau.fetch_add(1);
            if (tableau_index >= options.stock_order_tableaus) {
                break;
            }
            std::iota(shuffled.begin(), shuffled.end(), 0);
            std::mt19937_64 random(
                options.benchmark_seed + tableau_index * 0x9e3779b97f4a7c15ULL
            );
            std::shuffle(shuffled.begin(), shuffled.end(), random);
            std::copy(
                shuffled.begin(), shuffled.begin() + tableau_cards, deal.begin()
            );
            std::copy(
                shuffled.begin() + tableau_cards, shuffled.end(), stock.begin()
            );
            std::sort(stock.begin(), stock.end());

            bool first_outcome = false;
            bool have_outcome = false;
            do {
                std::copy(stock.begin(), stock.end(), deal.begin() + tableau_cards);
                bool outcome = solver.is_solvable(initial_state(deal, tableau_count));
                ++checked_deals;
                if (!have_outcome) {
                    first_outcome = outcome;
                    have_outcome = true;
                } else if (outcome != first_outcome) {
                    mismatch.store(true);
                    std::lock_guard<std::mutex> lock(mismatch_mutex);
                    std::ostringstream description;
                    description << "tableau_index=" << tableau_index << " stock=";
                    for (uint8_t card : stock) {
                        description << static_cast<int>(card) << ',';
                    }
                    description << " first=" << first_outcome << " current=" << outcome;
                    mismatch_description = description.str();
                    break;
                }
            } while (std::next_permutation(stock.begin(), stock.end()));
            ++checked_tableaus;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < options.threads; ++i) {
        threads.emplace_back(worker);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    std::cout << "stock_order_invariant " << (!mismatch.load() ? "true" : "false")
              << "\n";
    std::cout << "tableaus_checked " << checked_tableaus.load() << "\n";
    std::cout << "stock_orders_checked " << checked_deals.load() << "\n";
    std::cout << "elapsed_seconds " << elapsed << "\n";
    if (mismatch.load()) {
        std::cout << "counterexample " << mismatch_description << "\n";
        return 1;
    }
    return 0;
}

void generate_tableau_patterns_recursive(
    int n,
    int tableau_cards,
    std::array<int, kSuits>& counts,
    std::vector<uint8_t>& pattern,
    bool seen_suit_one,
    std::vector<std::vector<uint8_t>>& output
) {
    if (static_cast<int>(pattern.size()) == tableau_cards) {
        output.push_back(pattern);
        return;
    }
    for (int suit = 0; suit < kSuits; ++suit) {
        if (counts[suit] == n || (suit == 3 && !seen_suit_one)) {
            continue;
        }
        ++counts[suit];
        pattern.push_back(static_cast<uint8_t>(suit));
        generate_tableau_patterns_recursive(
            n,
            tableau_cards,
            counts,
            pattern,
            seen_suit_one || suit == 1,
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
    std::array<int, kSuits> counts{1, 0, 0, 0};
    std::vector<uint8_t> pattern{0};
    generate_tableau_patterns_recursive(
        n, tableau_cards, counts, pattern, false, patterns
    );
    return patterns;
}

void generate_partial_orders_recursive(
    int n,
    int count,
    std::vector<uint8_t>& current,
    std::array<bool, kSuits>& used,
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
        generate_partial_orders_recursive(n, count, current, used, output);
        current.pop_back();
        used[rank] = false;
    }
}

std::vector<std::vector<uint8_t>> partial_rank_orders(int n, int count) {
    std::vector<std::vector<uint8_t>> output;
    std::vector<uint8_t> current;
    std::array<bool, kSuits> used{};
    generate_partial_orders_recursive(n, count, current, used, output);
    return output;
}

struct TableauPatternWork {
    std::vector<uint8_t> pattern;
    std::array<int, kSuits> counts{};
    uint64_t combinations = 0;
    uint64_t byte_offset = 0;
};

int run_stock_collapsed(const Options& options) {
    int n = options.n;
    int tableau_count = default_tableau_count(n);
    int tableau_cards = tableau_count * (tableau_count + 1) / 2;
    auto patterns = generate_tableau_patterns(n, tableau_cards);
    std::array<std::vector<std::vector<uint8_t>>, kSuits + 1> rank_orders;
    for (int count = 0; count <= n; ++count) {
        rank_orders[count] = partial_rank_orders(n, count);
    }

    std::vector<TableauPatternWork> work;
    work.reserve(patterns.size());
    uint64_t total_deals = 0;
    uint64_t total_bytes = 0;
    for (std::vector<uint8_t>& pattern : patterns) {
        TableauPatternWork item;
        item.pattern = std::move(pattern);
        for (uint8_t suit : item.pattern) {
            ++item.counts[suit];
        }
        item.combinations = 1;
        for (int suit = 0; suit < kSuits; ++suit) {
            item.combinations *= rank_orders[item.counts[suit]].size();
        }
        if (item.combinations % 8 != 0) {
            throw std::runtime_error("tableau pattern is not byte aligned");
        }
        item.byte_offset = total_bytes;
        total_deals += item.combinations;
        total_bytes += item.combinations / 8;
        work.push_back(std::move(item));
    }

    uint64_t expected_deals = factorial(4 * n) /
        factorial(4 * n - tableau_cards) / 8;
    if (total_deals != expected_deals) {
        throw std::runtime_error("stock-collapsed deal count mismatch");
    }
    int output_file = open_sized_file(options.output, static_cast<off_t>(total_bytes));
    int completion_file = open_sized_file(
        options.completion, static_cast<off_t>(work.size())
    );
    if (output_file < 0 || completion_file < 0) {
        throw std::runtime_error("could not open stock-collapsed checkpoint files");
    }
    std::vector<uint8_t> completion(work.size());
    if (!read_exact(completion_file, completion.data(), completion.size(), 0)) {
        throw std::runtime_error("could not read stock-collapsed completion file");
    }

    uint64_t already_done = std::count(completion.begin(), completion.end(), 1);
    uint64_t already_deals = 0;
    for (size_t index = 0; index < completion.size(); ++index) {
        if (completion[index] == 1) {
            already_deals += work[index].combinations;
        }
    }
    std::atomic<uint64_t> next_pattern{0};
    std::atomic<uint64_t> claimed{0};
    std::atomic<uint64_t> done{already_done};
    std::atomic<uint64_t> deals_done{already_deals};
    std::atomic<uint64_t> expanded{0};
    std::mutex output_mutex;
    auto started = std::chrono::steady_clock::now();

    auto worker = [&]() {
        Solver solver(
            n,
            tableau_count,
            false,
            false,
            300,
            options.model_weights,
            options.max_recycles,
            options.exact_node_limit,
            options.stock_as_reserve,
            options.draw_count,
            options.allow_tableau_stack_splitting
        );
        std::vector<uint8_t> deal(4 * n);
        while (true) {
            uint64_t pattern_index = next_pattern.fetch_add(1);
            if (pattern_index >= work.size()) {
                break;
            }
            if (completion[pattern_index] == 1) {
                continue;
            }
            uint64_t claim = claimed.fetch_add(1);
            if (options.max_patterns > 0 && claim >= options.max_patterns) {
                break;
            }

            const TableauPatternWork& item = work[pattern_index];
            std::vector<uint8_t> outcomes(item.combinations / 8, 0);
            for (uint64_t combination = 0; combination < item.combinations; ++combination) {
                uint64_t remainder = combination;
                std::array<size_t, kSuits> rank_order_index{};
                for (int suit = kSuits - 1; suit >= 0; --suit) {
                    size_t order_count = rank_orders[item.counts[suit]].size();
                    rank_order_index[suit] = remainder % order_count;
                    remainder /= order_count;
                }
                std::array<size_t, kSuits> occurrence{};
                std::array<bool, kMaxCards> used_cards{};
                for (int position = 0; position < tableau_cards; ++position) {
                    int suit = item.pattern[position];
                    uint8_t rank_index = rank_orders[item.counts[suit]]
                        [rank_order_index[suit]][occurrence[suit]++];
                    uint8_t card = static_cast<uint8_t>(suit * n + rank_index);
                    deal[position] = card;
                    used_cards[card] = true;
                }
                int stock_position = tableau_cards;
                for (int card = 0; card < 4 * n; ++card) {
                    if (!used_cards[card]) {
                        deal[stock_position++] = static_cast<uint8_t>(card);
                    }
                }
                if (solver.is_solvable(initial_state(deal, tableau_count))) {
                    outcomes[combination / 8] |= static_cast<uint8_t>(
                        1U << (combination % 8)
                    );
                }
            }
            if (!write_exact(
                    output_file,
                    outcomes.data(),
                    outcomes.size(),
                    static_cast<off_t>(item.byte_offset))) {
                throw std::runtime_error("could not write stock-collapsed outcomes");
            }
            uint8_t complete = 1;
            if (!write_exact(completion_file, &complete, 1, pattern_index)) {
                throw std::runtime_error("could not write stock-collapsed completion");
            }
            completion[pattern_index] = 1;
            uint64_t current_deals = deals_done.fetch_add(item.combinations) +
                item.combinations;
            uint64_t current_done = done.fetch_add(1) + 1;
            if (current_done % 500 == 0 || current_done == work.size()) {
                double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started
                ).count();
                uint64_t newly_done = current_done - already_done;
                double patterns_per_second = newly_done / std::max(seconds, 0.001);
                uint64_t newly_classified = current_deals - already_deals;
                double deals_per_second = newly_classified / std::max(seconds, 0.001);
                double remaining = (total_deals - current_deals) /
                    std::max(deals_per_second, 0.001);
                std::lock_guard<std::mutex> lock(output_mutex);
                std::cout << "tableau_patterns " << current_done << "/" << work.size()
                          << " pattern_rate " << patterns_per_second
                          << "/s deal_rate " << deals_per_second << "/s eta "
                          << remaining / 3600.0 << "h\n" << std::flush;
            }
        }
        expanded.fetch_add(solver.expanded_positions);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < options.threads; ++i) {
        threads.emplace_back(worker);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    fsync(output_file);
    fsync(completion_file);
    close(output_file);
    close(completion_file);

    uint64_t completed = std::count(completion.begin(), completion.end(), 1);
    std::cout << "completed_tableau_patterns " << completed << "/" << work.size()
              << "\n";
    std::cout << "canonical_tableau_deals " << total_deals << "\n";
    std::cout << "outcome_bytes " << total_bytes << "\n";
    std::cout << "classified_tableau_deals " << deals_done.load() << "\n";
    std::cout << "expanded_positions " << expanded.load() << "\n";
    if (completed == work.size()) {
        std::vector<uint8_t> bits(total_bytes);
        int output_read = open(options.output.c_str(), O_RDONLY);
        read_exact(output_read, bits.data(), bits.size(), 0);
        close(output_read);
        uint64_t solvable = popcount_bytes(bits);
        std::cout << "solvable_tableau_deals " << solvable << "\n";
        std::cout << "unsolvable_tableau_deals " << total_deals - solvable << "\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        if (!options.verify_stock_order.empty()) {
            return verify_stock_order_invariance(options);
        }
        if (options.stock_order_tableaus > 0) {
            return check_stock_orders(options);
        }
        if (options.collapse_stock_order) {
            return run_stock_collapsed(options);
        }
        if (options.benchmark_deals > 0) {
            return run_benchmark(options);
        }
        int n = options.n;
        int tableau_count = default_tableau_count(n);
        auto patterns = generate_patterns(n);
        auto ranks = rank_permutations(n);
        uint64_t expected_patterns = factorial(4 * n) / (factorial(n) * factorial(n) *
            factorial(n) * factorial(n)) / 8;
        if (patterns.size() != expected_patterns) {
            throw std::runtime_error("canonical suit pattern count mismatch");
        }

        uint64_t deals_per_pattern = 1;
        for (int suit = 0; suit < kSuits; ++suit) {
            deals_per_pattern *= ranks.size();
        }
        if (deals_per_pattern % 8 != 0) {
            throw std::runtime_error("deals per pattern must be byte aligned");
        }
        uint64_t bytes_per_pattern = deals_per_pattern / 8;
        uint64_t total_deals = patterns.size() * deals_per_pattern;
        uint64_t total_bytes = total_deals / 8;

        int output_file = open_sized_file(options.output, static_cast<off_t>(total_bytes));
        int completion_file = open_sized_file(
            options.completion, static_cast<off_t>(patterns.size())
        );
        if (output_file < 0 || completion_file < 0) {
            throw std::runtime_error("could not open checkpoint files");
        }

        std::vector<uint8_t> completion(patterns.size());
        if (!read_exact(completion_file, completion.data(), completion.size(), 0)) {
            throw std::runtime_error("could not read completion checkpoint");
        }
        uint64_t already_done = std::count(completion.begin(), completion.end(), 1);
        std::atomic<uint64_t> next_pattern{0};
        std::atomic<uint64_t> claimed{0};
        std::atomic<uint64_t> done{already_done};
        std::atomic<uint64_t> solvable_new{0};
        std::atomic<uint64_t> expanded{0};
        std::mutex output_mutex;
        auto started = std::chrono::steady_clock::now();
        std::atomic<uint64_t> model_attempts{0};
        std::atomic<uint64_t> model_wins{0};
        std::atomic<uint64_t> model_steps{0};
        std::atomic<uint64_t> exact_fallbacks{0};

        auto worker = [&]() {
            Solver solver(
                n,
                tableau_count,
                options.model_enabled,
                options.model_only,
                options.model_max_steps,
                options.model_weights,
                options.max_recycles,
                options.exact_node_limit,
                false,
                options.draw_count,
                options.allow_tableau_stack_splitting
            );
            std::vector<uint8_t> deal(4 * n);
            while (true) {
                uint64_t pattern_index = next_pattern.fetch_add(1);
                if (pattern_index >= patterns.size()) {
                    break;
                }
                if (completion[pattern_index] == 1) {
                    continue;
                }
                uint64_t claim = claimed.fetch_add(1);
                if (options.max_patterns > 0 && claim >= options.max_patterns) {
                    break;
                }

                std::vector<uint8_t> outcome(bytes_per_pattern, 0);
                uint64_t pattern_solvable = 0;
                const std::vector<uint8_t>& pattern = patterns[pattern_index];
                for (uint64_t combination = 0; combination < deals_per_pattern; ++combination) {
                    uint64_t remainder = combination;
                    std::array<size_t, kSuits> rank_order{};
                    for (int suit = kSuits - 1; suit >= 0; --suit) {
                        rank_order[suit] = remainder % ranks.size();
                        remainder /= ranks.size();
                    }
                    std::array<size_t, kSuits> occurrence{};
                    for (size_t position = 0; position < pattern.size(); ++position) {
                        int suit = pattern[position];
                        uint8_t rank_index = ranks[rank_order[suit]][occurrence[suit]++];
                        deal[position] = static_cast<uint8_t>(suit * n + rank_index);
                    }

                    bool solvable = solver.is_solvable(
                        initial_state(deal, tableau_count)
                    );
                    if (solvable) {
                        outcome[combination / 8] |= static_cast<uint8_t>(
                            1U << (combination % 8)
                        );
                        ++pattern_solvable;
                    }
                }

                off_t offset = static_cast<off_t>(pattern_index * bytes_per_pattern);
                if (!write_exact(output_file, outcome.data(), outcome.size(), offset)) {
                    throw std::runtime_error("could not write outcome checkpoint");
                }
                uint8_t complete = 1;
                if (!write_exact(completion_file, &complete, 1, pattern_index)) {
                    throw std::runtime_error("could not write completion checkpoint");
                }
                completion[pattern_index] = 1;
                solvable_new.fetch_add(pattern_solvable);
                uint64_t current_done = done.fetch_add(1) + 1;
                if (current_done % 250 == 0 || current_done == patterns.size()) {
                    auto now = std::chrono::steady_clock::now();
                    double seconds = std::chrono::duration<double>(now - started).count();
                    uint64_t newly_done = current_done - already_done;
                    double patterns_per_second = newly_done / std::max(seconds, 0.001);
                    double remaining = (patterns.size() - current_done) /
                        std::max(patterns_per_second, 0.001);
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cout << "patterns " << current_done << "/" << patterns.size()
                              << " rate " << patterns_per_second << "/s eta "
                              << remaining / 60.0 << "m\n" << std::flush;
                }
            }
            expanded.fetch_add(solver.expanded_positions);
            model_attempts.fetch_add(solver.model_attempts);
            model_wins.fetch_add(solver.model_wins);
            model_steps.fetch_add(solver.model_steps);
            exact_fallbacks.fetch_add(solver.exact_fallbacks);
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < options.threads; ++i) {
            threads.emplace_back(worker);
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
        fsync(output_file);
        fsync(completion_file);
        close(output_file);
        close(completion_file);

        std::vector<uint8_t> final_completion(patterns.size());
        int completion_read = open(options.completion.c_str(), O_RDONLY);
        read_exact(completion_read, final_completion.data(), final_completion.size(), 0);
        close(completion_read);
        uint64_t completed = std::count(final_completion.begin(), final_completion.end(), 1);
        std::cout << "completed_patterns " << completed << "/" << patterns.size() << "\n";
        std::cout << "newly_solvable " << solvable_new.load() << "\n";
        std::cout << "model_attempts " << model_attempts.load() << "\n";
        std::cout << "model_wins " << model_wins.load() << "\n";
        std::cout << "model_steps " << model_steps.load() << "\n";
        std::cout << "exact_fallbacks " << exact_fallbacks.load() << "\n";
        std::cout << "expanded_positions " << expanded.load() << "\n";

        if (completed == patterns.size()) {
            std::vector<uint8_t> bits(total_bytes);
            int output_read = open(options.output.c_str(), O_RDONLY);
            read_exact(output_read, bits.data(), bits.size(), 0);
            close(output_read);
            uint64_t solvable = popcount_bytes(bits);
            std::cout << "canonical_deals " << total_deals << "\n";
            std::cout << "solvable_deals " << solvable << "\n";
            std::cout << "unsolvable_deals " << total_deals - solvable << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
