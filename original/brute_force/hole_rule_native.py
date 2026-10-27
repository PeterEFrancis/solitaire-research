"""Build an isolated last-empty-column experiment without editing native_solver.cpp.

From original/::

    python3 -m brute_force.hole_rule_native --output /private/tmp/solitaire-hole-rule

The resulting executable adds --hole-penalty FLOAT and --preserve-last-hole.
Active interventions require a draw-three/unlimited model-only benchmark. They
alter candidate selection only: successor legality, feature values, order,
cycle suppression, and exact search remain those of the source engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


SOURCE = Path(__file__).with_name("native_solver.cpp")


def replace_once(source: str, anchor: str, replacement: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise ValueError(f"native source anchor must occur once, found {count}: {anchor[:90]!r}")
    return source.replace(anchor, replacement, 1)


def patched_source(source: str) -> str:
    if "g_hole_penalty" in source or "consumes_last_hole" in source:
        raise ValueError("source already contains the experimental hole rule")
    source = replace_once(source, "#include <chrono>\n", "#include <chrono>\n#include <cmath>\n#include <iomanip>\n")
    source = replace_once(source, "struct Key {\n", r'''// Isolated policy intervention; globals become read-only before workers start.
double g_hole_penalty = 0.0;
bool g_preserve_last_hole = false;

int count_holes(const State& state, int tableau_count) {
    int holes = 0;
    for (int column = 0; column < tableau_count; ++column) {
        holes += state.tableau[column].size == 0;
    }
    return holes;
}

bool consumes_last_hole(const State& before, const State& after, int tableau_count) {
    return count_holes(before, tableau_count) == 1 &&
        count_holes(after, tableau_count) == 0;
}

struct Key {
''')
    source = replace_once(source, "            Key choice_key{};\n", "            Key choice_key{};\n            double choice_score = 0.0;\n")
    source = replace_once(source, r'''                if (choice == nullptr || move.model_score > choice->model_score) {
                    choice = &move;
                    choice_key = next_key;
                }
''', r'''                double candidate_score = move.model_score;
                if ((g_preserve_last_hole || g_hole_penalty > 0.0) &&
                    consumes_last_hole(state, move.state, tableau_count_)) {
                    if (g_preserve_last_hole) {
                        continue;
                    }
                    candidate_score -= g_hole_penalty;
                }
                if (choice == nullptr || candidate_score > choice_score) {
                    choice = &move;
                    choice_score = candidate_score;
                    choice_key = next_key;
                }
''')
    source = replace_once(source, "Options parse_options(int argc, char** argv) {\n    Options options;\n", "Options parse_options(int argc, char** argv) {\n    Options options;\n    g_hole_penalty = 0.0;\n    g_preserve_last_hole = false;\n")
    source = replace_once(source, '        if (argument == "--disable-model") {\n', '        if (argument == "--preserve-last-hole") {\n            g_preserve_last_hole = true;\n            continue;\n        }\n        if (argument == "--disable-model") {\n')
    source = replace_once(source, '        if (argument == "--n") {\n', r'''        if (argument == "--hole-penalty") {
            size_t consumed = 0;
            g_hole_penalty = std::stod(value, &consumed);
            if (consumed != value.size() || !std::isfinite(g_hole_penalty) ||
                g_hole_penalty < 0.0) {
                throw std::runtime_error("--hole-penalty requires a finite nonnegative number");
            }
        } else if (argument == "--n") {
''')
    source = replace_once(source, "    if (options.draw_count != 1 && options.draw_count != 3) {\n", r'''    if ((g_preserve_last_hole || g_hole_penalty > 0.0) &&
        (options.benchmark_deals == 0 || !options.model_only || !options.model_enabled ||
         options.draw_count != 3 || options.max_recycles != -1 ||
         options.stock_order_tableaus != 0 || !options.verify_stock_order.empty() ||
         options.collapse_stock_order || options.stock_as_reserve)) {
        throw std::runtime_error(
            "hole interventions require a draw-three/unlimited model-only benchmark"
        );
    }
    if (options.draw_count != 1 && options.draw_count != 3) {
''')
    source = replace_once(source, '    std::cout << "benchmark_deals " << options.benchmark_deals << "\\n";\n', r'''    if (g_preserve_last_hole || g_hole_penalty > 0.0) {
        std::streamsize old_precision = std::cout.precision();
        std::cout << "hole_penalty " << std::setprecision(17) << g_hole_penalty << "\n";
        std::cout.precision(old_precision);
        std::cout << "preserve_last_hole " << (g_preserve_last_hole ? 1 : 0) << "\n";
    }
    std::cout << "benchmark_deals " << options.benchmark_deals << "\n";
''')
    return source


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiler_spec(cxx: str | None = None, cxxflags: str | None = None) -> tuple[list[str], list[str]]:
    compiler = shlex.split(cxx or os.environ.get("CXX", "clang++"))
    if not compiler or not shutil.which(compiler[0]):
        raise ValueError("C++ compiler unavailable; specify --cxx or CXX")
    # Keep the clang++/g++ launcher name: resolving its symlink can change
    # the driver's automatic C++ standard-library linkage.
    compiler[0] = str(Path(shutil.which(compiler[0])).absolute())
    supplied = cxxflags if cxxflags is not None else os.environ.get("HOLE_RULE_CXXFLAGS", "")
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-pthread", *shlex.split(supplied)]
    sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk")
    if not supplied and platform.system() == "Darwin" and sdk.is_dir():
        flags.extend(["-isysroot", str(sdk), "-isystem", str(sdk / "usr/include/c++/v1")])
    return compiler, flags


def build_binary(
    output: Path,
    *,
    manifest_path: Path | None = None,
    source_path: Path = SOURCE,
    cxx: str | None = None,
    cxxflags: str | None = None,
) -> dict[str, Any]:
    output, source_path = Path(output).resolve(), Path(source_path).resolve()
    generated_path = Path(str(output) + ".cpp")
    manifest_path = Path(manifest_path).resolve() if manifest_path is not None else Path(str(output) + ".build.json")
    if len({output, generated_path, manifest_path, source_path}) != 4:
        raise ValueError("binary, generated source, manifest, and original source must have distinct paths")
    original = source_path.read_bytes()
    generated = patched_source(original.decode("utf-8"))
    compiler, flags = compiler_spec(cxx, cxxflags)
    version = subprocess.run([*compiler, "--version"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="solitaire-hole-build-", dir=output.parent) as directory:
        temporary_source = Path(directory) / "native_solver.holes.cpp"
        temporary_binary = Path(directory) / "native_solver.holes"
        temporary_source.write_text(generated, encoding="utf-8")
        command = [*compiler, *flags, str(temporary_source), "-o", str(temporary_binary)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
        if source_path.read_bytes() != original:
            raise ValueError("original native source changed during the build")
        temporary_binary.replace(output)
        temporary_source.replace(generated_path)
    manifest = {
        "schema_version": 1,
        "experiment": "last-empty-tableau-column policy intervention",
        "predicate": "empty_columns(before) == 1 and empty_columns(after) == 0",
        "restriction": "Active interventions require draw-three/unlimited model-only benchmarks; no change to legal successors or exact search.",
        "options": {"hole_penalty_default": 0.0, "preserve_last_hole_default": False},
        "source": {"path": str(source_path), "sha256": hashlib.sha256(original).hexdigest()},
        "generated_source": {"path": str(generated_path), "sha256": sha256(generated_path)},
        "binary": {"path": str(output), "sha256": sha256(output)},
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "compiler": {"command": compiler, "version": version,
                     "executable_sha256": sha256(Path(compiler[0])),
                     "version_sha256": hashlib.sha256(version.encode("utf-8")).hexdigest()},
        "flags": flags,
        "compile_command": [*compiler, *flags, str(generated_path), "-o", str(output)],
        "platform": platform.platform(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = Path(str(manifest_path) + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "solitaire-hole-rule")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cxx")
    parser.add_argument("--cxxflags")
    args = parser.parse_args(argv)
    manifest = build_binary(args.output, manifest_path=args.manifest, cxx=args.cxx, cxxflags=args.cxxflags)
    print(f"Built isolated hole-rule executable: {manifest['binary']['path']}")
    print(f"Binary SHA-256: {manifest['binary']['sha256']}")
    print(f"Manifest: {args.manifest or str(args.output) + '.build.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
