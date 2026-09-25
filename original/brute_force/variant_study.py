"""Train, validate, freeze, and independently evaluate Klondike rule variants.

Run from original/. Each split uses common indexed deals across policies and
variants. The test split never chooses weights or portfolio members. Portfolios
are full-information restarts: each member plays the same initial deal, then a
winning trajectory (or the largest foundation count) is selected.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import platform
import subprocess
import tempfile
import threading

from solitaire.player import PARAMETER_NAMES
from .compare_policies import paired_statistics, sha256_file, validate_outcomes, wilson_interval, write_report
from .tune_model import DEFAULT_STEPS

RESULTS = Path(__file__).resolve().parent / "results"
PROFILES = {
    "draw1_limited": (1, 3, True),
    "draw1_unlimited": (1, None, True),
    "draw3_limited": (3, 3, True),
    "draw3_unlimited": (3, None, True),
    "vegas": (1, 0, False),
}
SIMPLE = ("safe_foundation", "reveal_hidden", "tableau_build", "stock_action",
          "waste_to_tableau", "reveal_depth", "recycle", "productive_stack_length")
TRAIN_SEED, VALIDATION_SEED, TEST_SEED = 2026092601, 2026092602, 2026092699


def configuration(profile):
    draw, recycles, split = PROFILES[profile]
    return dict(n=13, k=2, t=7, draw_count=draw, max_recycles=recycles,
                allow_tableau_stack_splitting=split)


def vector(weights):
    return tuple(float(weights.get(name, 0.0)) for name in PARAMETER_NAMES)


def named(weights):
    return dict(zip(PARAMETER_NAMES, weights))


def normalize(weights, profile):
    result = list(weights)
    if PROFILES[profile][1] is None:
        for name in ("recycle_pressure", "waste_play_recycle_pressure"):
            result[PARAMETER_NAMES.index(name)] = 0.0
    return tuple(result)


def evaluate(binary, profile, weights, deals, seed, threads, max_steps, *, record=False):
    draw, recycles, split = PROFILES[profile]
    with tempfile.TemporaryDirectory(prefix="solitaire-variant-") as directory:
        outcome_path, foundation_path = Path(directory)/"wins.u8", Path(directory)/"foundations.u8"
        command = [str(binary), "--n", "13", "--draw-count", str(draw),
                   "--max-recycles", str(-1 if recycles is None else recycles),
                   "--model-only", "--model-max-steps", str(max_steps),
                   "--model-weights", ",".join(format(x, ".17g") for x in weights),
                   "--benchmark-deals", str(deals), "--benchmark-seed", str(seed),
                   "--threads", str(threads)]
        if not split:
            command.append("--no-tableau-splitting")
        if record:
            command += ["--benchmark-outcomes", str(outcome_path),
                        "--benchmark-foundations", str(foundation_path)]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        fields = dict(line.split(" ", 1) for line in completed.stdout.splitlines())
        if int(fields["benchmark_deals"]) != deals or int(fields["exact_fallbacks"]) != 0:
            raise ValueError("wrong native benchmark mode")
        row = {"deals": deals, "wins": int(fields["model_wins"]),
               "foundation_cards": int(fields["model_foundation_cards"]),
               "foundation_cards_squared": int(fields["model_foundation_cards_squared"]),
               "steps": int(fields["model_steps"]), "cutoffs": int(fields["model_cutoffs"])}
        outcomes = outcome_path.read_bytes() if record else b""
        foundations = foundation_path.read_bytes() if record else b""
        if record:
            validate_outcomes(outcomes, deals, row["wins"])
            if (len(foundations) != deals or any(x > 52 for x in foundations)
                    or sum(foundations) != row["foundation_cards"]
                    or sum(x*x for x in foundations) != row["foundation_cards_squared"]
                    or any(bool(w) != (f == 52) for w, f in zip(outcomes, foundations))):
                raise ValueError("foundation outcomes disagree with aggregate results")
        return row, outcomes, foundations


def objective(row, profit=False):
    if profit:
        return row["foundation_cards"], row["wins"], -row["steps"]
    return row["wins"], row["foundation_cards"], -row["steps"]


def statistics(row):
    result = dict(row)
    n = row["deals"]
    mean = row["foundation_cards"] / n
    variance = max(0, row["foundation_cards_squared"] / n - mean * mean)
    margin = 1.96 * math.sqrt(variance / max(1, n-1))
    result.update(win_rate=row["wins"]/n,
                  win_rate_wilson_95_ci=list(wilson_interval(row["wins"], n)),
                  mean_foundation_cards=mean,
                  mean_foundation_cards_95_ci=[mean-margin, mean+margin],
                  mean_net_dollars=5*mean-52,
                  mean_net_dollars_95_ci=[5*(mean-margin)-52, 5*(mean+margin)-52])
    if "steps" in row:
        result["mean_steps"] = row["steps"]/n
    return result


def paired(outcomes, baseline):
    return paired_statistics(sum(a == 1 and b == 0 for a,b in zip(outcomes,baseline)),
                             sum(a == 0 and b == 1 for a,b in zip(outcomes,baseline)), len(outcomes))


def train_profile(args, profile, stage8, simple, visible_names):
    rng = random.Random(2026092600 + list(PROFILES).index(profile))
    cache, history = {}, []
    def train(weights):
        weights = normalize(weights, profile)
        if weights not in cache:
            row, _, _ = evaluate(args.binary, profile, weights, args.train_deals,
                                 TRAIN_SEED, args.threads, args.max_steps)
            cache[weights] = row
            history.append({"parameters": named(weights), **row})
        return cache[weights]

    initial_full = normalize(stage8, profile)
    initial_simple = normalize(simple, profile)
    initial_visible = normalize(tuple(w if name in visible_names else 0.0
                                      for name,w in zip(PARAMETER_NAMES,stage8)), profile)
    selected, validation_records, full_finalists = {}, {}, []
    families = ["full", "visible", "simple_eight"]
    if profile == "vegas":
        families.append("profit")
    for family in families:
        profit = family == "profit"
        allowed = (list(SIMPLE) if family == "simple_eight" else
                   list(visible_names) if family == "visible" else list(PARAMETER_NAMES))
        allowed_indices = [i for i,n in enumerate(PARAMETER_NAMES) if n in allowed]
        initial = initial_simple if family == "simple_eight" else initial_visible if family == "visible" else initial_full
        pool = {initial}
        if family in ("full", "profit"):
            pool.update((initial_visible, initial_simple))
        if family == "visible":
            pool.add(initial_simple)
        # Rule-sensitive starting points, followed by a coordinate sweep and
        # independent sparse mutations. Every choice below uses training only.
        for stock, recycle, foundation in ((-2,-3,0),(-0.5,0,0),(-1,0,2),(-1,-2,5)):
            changed = list(initial)
            for name,value in (("stock_action",stock),("recycle",recycle),("foundation_move",foundation)):
                if name in allowed:
                    changed[PARAMETER_NAMES.index(name)] = value
            pool.add(normalize(changed,profile))
        best = max(pool, key=lambda w: objective(train(w), profit))
        for i in allowed_indices:
            step = max(DEFAULT_STEPS[i]*2, abs(best[i])*0.35)
            if family == "simple_eight":
                step = max(step, 0.5)
            candidates = [best]
            for delta in (-step, step):
                changed = list(best); changed[i] += delta
                if family == "simple_eight":
                    changed[i] = round(changed[i]*2)/2
                candidate = normalize(changed,profile)
                pool.add(candidate); candidates.append(candidate)
            best = max(candidates, key=lambda w: objective(train(w), profit))
        for _ in range(args.mutations):
            elite = sorted(pool, key=lambda w: objective(train(w),profit), reverse=True)[:4]
            changed = list(rng.choice(elite))
            for i in rng.sample(allowed_indices, min(3,len(allowed_indices))):
                changed[i] += rng.gauss(0, max(DEFAULT_STEPS[i],abs(changed[i])*0.2))
                if family == "simple_eight":
                    changed[i] = round(changed[i]*2)/2
            candidate = normalize(changed,profile); pool.add(candidate); train(candidate)
        finalists = sorted(pool,key=lambda w:objective(train(w),profit),reverse=True)[:args.finalists]
        if initial not in finalists:
            finalists.append(initial)
        validation = []
        for weights in finalists:
            row,_,_ = evaluate(args.binary,profile,weights,args.validation_deals,
                              VALIDATION_SEED,args.threads,args.max_steps)
            validation.append({"parameters":named(weights),**row})
        validation.sort(key=lambda row:objective(row,profit),reverse=True)
        selected[family] = vector(validation[0]["parameters"])
        validation_records[family] = validation
        if family == "full":
            full_finalists = [vector(row["parameters"]) for row in validation[:3]]
        best_row = validation[0]
        print(f"{profile} {family}: validation {best_row['wins']}/{best_row['deals']} "
              f"({best_row['wins']/best_row['deals']:.3%}); foundations={best_row['foundation_cards']/best_row['deals']:.3f}",flush=True)

    # Freeze all portfolio members using validation only. The bank includes
    # diverse observable policies as alternative full-information trajectories.
    portfolio = list(dict.fromkeys(full_finalists + [initial_full,selected["visible"],selected["simple_eight"]]))
    training = {"config":configuration(profile),"training_evaluations":len(cache),
                "history":history,"validation":validation_records}
    frozen = {"config":configuration(profile),"parameters":named(selected["full"]),
              "policies":{name:named(w) for name,w in selected.items()},
              "portfolio":[named(w) for w in portfolio]}
    return training,frozen


def confirm_profile(args,profile,frozen,stage8,simple,output_dir):
    policies = {"stage8_baseline":normalize(stage8,profile),
                "simple_eight_baseline":normalize(simple,profile)}
    policies.update({name:vector(w) for name,w in frozen["policies"].items()})
    for i,w in enumerate(frozen["portfolio"]):
        policies[f"portfolio_{i}"] = vector(w)
    cache, evaluations, raw = {}, {}, {}
    for name,weights in policies.items():
        if weights not in cache:
            cache[weights] = evaluate(args.binary,profile,weights,args.test_deals,
                                     TEST_SEED,args.threads,args.max_steps,record=True)
        row,wins,foundations = cache[weights]
        entry = statistics(row)
        for kind,data in (("wins",wins),("foundations",foundations)):
            filename = f"{profile}.{name}.{kind}.u8.gz"
            compressed = gzip.compress(data,mtime=0)
            (output_dir/filename).write_bytes(compressed)
            entry[kind+"_file"] = str((output_dir/filename).relative_to(RESULTS))
            entry[kind+"_sha256"] = hashlib.sha256(data).hexdigest()
        entry["paired_vs_stage8"] = paired(wins,raw["stage8_baseline"][0] if raw else wins)
        evaluations[name],raw[name] = entry,(wins,foundations)
        print(f"{profile} {name}: test {entry['win_rate']:.4%}",flush=True)
    member_names = [f"portfolio_{i}" for i in range(len(frozen["portfolio"]))]
    members = [raw[name] for name in member_names]
    wins = bytes(max(values) for values in zip(*(m[0] for m in members)))
    foundation = bytes(max(values) for values in zip(*(m[1] for m in members)))
    bank = statistics(dict(deals=args.test_deals,wins=sum(wins),foundation_cards=sum(foundation),
                           foundation_cards_squared=sum(x*x for x in foundation)))
    bank.update(members=member_names,attempts_per_deal=len(members),
                paired_vs_stage8=paired(wins,raw["stage8_baseline"][0]),
                paired_vs_selected_full=paired(wins,raw["full"][0]))
    for kind,data in (("wins",wins),("foundations",foundation)):
        filename=f"{profile}.portfolio.{kind}.u8.gz"
        (output_dir/filename).write_bytes(gzip.compress(data,mtime=0))
        bank[kind+"_file"]=str((output_dir/filename).relative_to(RESULTS))
        bank[kind+"_sha256"]=hashlib.sha256(data).hexdigest()
    evaluations["portfolio"] = bank
    evaluations["simple_eight"]["paired_vs_simple_baseline"] = paired(raw["simple_eight"][0],raw["simple_eight_baseline"][0])
    return {"config":configuration(profile),"evaluations":evaluations}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,default=Path("brute_force/native_solver.variants"))
    parser.add_argument("--output-prefix",default="variant-study")
    parser.add_argument("--train-deals",type=int,default=5000)
    parser.add_argument("--validation-deals",type=int,default=50000)
    parser.add_argument("--test-deals",type=int,default=250000)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--jobs",type=int,default=3)
    parser.add_argument("--max-steps",type=int,default=1000)
    parser.add_argument("--mutations",type=int,default=24)
    parser.add_argument("--finalists",type=int,default=4)
    parser.add_argument("--profiles",nargs="+",choices=list(PROFILES),default=list(PROFILES))
    parser.add_argument("--confirm-only",action="store_true")
    args=parser.parse_args()
    if any(getattr(args,x)<1 for x in ("train_deals","validation_deals","test_deals","threads","jobs","max_steps","finalists")) or args.mutations<0:
        parser.error("counts must be positive; mutations nonnegative")
    if Path(args.output_prefix).name!=args.output_prefix:
        parser.error("output-prefix must be a filename prefix")
    args.binary=args.binary.resolve()
    stage8=vector(json.loads((RESULTS/"k2_n13_t7.tuning-stage8.json").read_text())["selected_parameters"])
    simple=vector(json.loads((RESULTS/"k2_n13_t7.human-frozen-policies.json").read_text())["simple_eight"])
    visible=json.loads((RESULTS/"human-strategy-candidates.json").read_text())["audit"]["current_visible_features"]
    prefix=RESULTS/args.output_prefix
    report_path=Path(str(prefix)+".confirmation.json")
    output_dir=RESULTS/(args.output_prefix+"-outcomes")
    if report_path.exists() or output_dir.exists():
        raise FileExistsError("confirmation artifacts already exist; use a new study prefix")
    protocol={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),
              "options":{**vars(args),"binary":str(args.binary)},
              "seeds":{"train":TRAIN_SEED,"validation":VALIDATION_SEED,"test":TEST_SEED},
              "method":"Rule-specific coordinate search plus sparse mutations on common training deals; validation chooses fixed single policies and portfolio members; every frozen policy is reported on the untouched test split.",
              "information":"Full models and restart portfolios inspect unknown cards. Visible and eight-term scorers exclude unknown-card identities and waste history. All computer policies use complete position keys for cycle avoidance.",
              "objectives":{"full":"complete wins","visible":"complete wins","simple_eight":"complete wins","profit":"foundation cards; Vegas only"},
              "portfolio":"Full-information planning with up to six frozen policies restarted from the same deal, accepting any winning trajectory; maximum foundation count among trajectories on failures. More computation than a single policy.",
              "limitations":"Best among candidates searched, not globally optimal. Fixed move budget. Test confidence intervals describe sampling, not model-class uncertainty. $5 per foundation card minus $52 is an accounting convention for the explicitly defined Vegas rules.",
              "binary_sha256":sha256_file(args.binary),
              "python_version":platform.python_version(),
              "baseline_parameters":{"stage8":named(stage8),"simple_eight":named(simple)},
              "baseline_source_sha256":{
                  "stage8":sha256_file(RESULTS/"k2_n13_t7.tuning-stage8.json"),
                  "simple_eight":sha256_file(RESULTS/"k2_n13_t7.human-frozen-policies.json")},
              "native_source_sha256":sha256_file(Path(__file__).with_name("native_solver.cpp")),
              "experiment_source_sha256":sha256_file(Path(__file__))}
    frozen_path=Path(str(prefix)+".selected-policies.json")
    training_path=Path(str(prefix)+".training.json")
    protocol_path=Path(str(prefix)+".protocol.json")
    if args.confirm_only:
        frozen=json.loads(frozen_path.read_text())
        saved=json.loads(protocol_path.read_text())
        if not frozen.get("complete"):
            raise ValueError("policies have not all been frozen")
        for key in ("binary_sha256","native_source_sha256","experiment_source_sha256",
                    "baseline_source_sha256","baseline_parameters","seeds"):
            if saved[key]!=protocol[key]:
                raise ValueError(f"confirmation provenance differs: {key}")
        for key in ("max_steps","test_deals","profiles"):
            if saved["options"][key]!=protocol["options"][key]:
                raise ValueError(f"confirmation settings differ: {key}")
    else:
        if frozen_path.exists() or training_path.exists():
            raise FileExistsError("choose a new output prefix to avoid overwriting a study")
        write_report(protocol_path,protocol)
        training={"schema_version":1,"complete":False,"variants":{}}
        frozen={"schema_version":1,"complete":False,"variants":{}}
        lock=threading.Lock()
        def train_one(profile):
            tr,fr=train_profile(args,profile,stage8,simple,visible)
            with lock:
                training["variants"][profile]=tr; frozen["variants"][profile]=fr
                write_report(training_path,training); write_report(frozen_path,frozen)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(train_one,args.profiles))
        training["complete"]=frozen["complete"]=True
        write_report(training_path,training);write_report(frozen_path,frozen)
    output_dir.mkdir()
    report={"schema_version":1,"complete":False,"protocol":protocol_path.name,
            "frozen_policies_sha256":sha256_file(frozen_path),
            "evaluation_options":{"deals":args.test_deals,"seed":TEST_SEED,
                                  "max_steps":args.max_steps,"threads":args.threads},
            "variants":{}}
    lock=threading.Lock()
    def confirm_one(profile):
        row=confirm_profile(args,profile,frozen["variants"][profile],stage8,simple,output_dir)
        with lock:
            report["variants"][profile]=row;write_report(report_path,report)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(confirm_one,args.profiles))
    report["complete"]=True
    report["finished_at"]=datetime.now(timezone.utc).isoformat()
    write_report(report_path,report)


if __name__=="__main__":
    main()
