#!/usr/bin/env python3
"""Build the portable C++ player and record the exact build provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"original/brute_force/native_solver.variants")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    compiler = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        parser.error("install a C++17 compiler (clang++ or g++)")
    source = ROOT/"original/brute_force/native_solver.cpp"
    command = [compiler, "-std=c++17", "-O3", "-DNDEBUG", "-pthread"]
    if platform.system() == "Darwin":
        sdk = Path(subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip())
        command += ["-isysroot", str(sdk)]
        headers = sdk/"usr/include/c++/v1"
        if headers.is_dir():
            command += ["-isystem", str(headers)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command += [str(source), "-o", str(args.output)]
    subprocess.run(command, check=True)
    manifest = {"command": command, "platform": platform.platform(),
                "compiler_version": subprocess.check_output([compiler,"--version"],text=True).strip(),
                "source_sha256":digest(source),"binary_sha256":digest(args.output)}
    if args.manifest:
        args.manifest.parent.mkdir(parents=True,exist_ok=True)
        args.manifest.write_text(json.dumps(manifest,indent=2)+"\n")
    print("Built",args.output)
    print("SHA-256",manifest["binary_sha256"])


if __name__ == "__main__":
    main()
