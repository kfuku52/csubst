"""Diagnostic: all_candidates bypasses foreground-only candidate filtering.

Usage: python reproduce.py marginal|joint OUTDIR normal|all_candidates|bundled|independent
The all_candidates override is in-memory only; it does not modify package code.
"""

import sys
import runpy
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    mode, out = sys.argv[1:3]
    out = Path(out)
    scenario = sys.argv[3] if len(sys.argv) > 3 else "bundled"
    helper = runpy.run_path(str(root / ".github/scripts/benchmark_endpoints.py"))
    command = helper["cli_args"]("PEPC", mode, 64, out)
    command[command.index("--max_arity") + 1] = "6"
    command += ["--cutoff_stat", "OCNany2spe,2.0|omegaCany2spe,5.0"]
    if scenario == "bundled":
        command += [
            "--foreground",
            str(root / "csubst/dataset/PEPC.foreground.txt"),
            "--fg_format",
            "1",
        ]
    elif scenario == "independent":
        fg = Path(str(out) + ".foreground.tsv")
        names = [
            line.split()[1]
            for line in (root / "csubst/dataset/PEPC.foreground.txt")
            .read_text()
            .splitlines()
        ]
        fg.write_text(
            "name\tC4\n" + "".join(f"{name}\t{i + 1}\n" for i, name in enumerate(names))
        )
        command += ["--foreground", str(fg), "--fg_format", "2"]
    elif scenario == "normal":
        pass
    elif scenario == "all_candidates":
        from csubst import combination

        original_combinations = combination.get_node_combinations

        def all_candidates(*args, **kwargs):
            if kwargs.get("cb_passed") is not None:
                kwargs["cb_all"] = True
            return original_combinations(*args, **kwargs)

        combination.get_node_combinations = all_candidates
    else:
        raise ValueError(scenario)
    from csubst import tsv

    original = tsv.write_dataframe

    def capture(df, path, **kwargs):
        if Path(path).name.startswith("csubst_cb_"):
            df.to_pickle(str(path) + ".unrounded.pkl")
        return original(df, path, **kwargs)

    tsv.write_dataframe = capture
    Path(str(out) + ".command.json").write_text(json.dumps(command, indent=2))
    sys.argv = command
    runpy.run_module("csubst", run_name="__main__")


if __name__ == "__main__":
    main()
