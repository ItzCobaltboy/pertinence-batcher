"""
Whole pipeline, one command:

    python run_all.py            # full paper budget (pop 50 x 50 generations)
    python run_all.py --smoke    # tiny budget into results_smoke/, minutes
    python run_all.py --from search   # resume from a stage

Stages, each its own process (dispatcher/ and dispatcher_analysis/ have
same-named modules, e.g. embeddings.py, so they can't share one interpreter):

  label     label_data.py                 ground-truth CSVs from yolo-analysis results
  images    download_images.py            fetch any COCO images not already on disk
  costs     measure_costs.py              GFLOPs of the pool + extractor -> data/model_costs.json
  search    run_dispatcher.py             embeddings (cached) + NSGA-II + Pareto FC weights
  analysis  run_dispatcher_analysis.py    test + final_val evaluation, CSVs and plots

label and costs are skipped when their outputs already exist (they're
deterministic); pass --redo to force them.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    ("label", "label_data.py"),
    ("images", "download_images.py"),
    ("costs", "measure_costs.py"),
    ("search", "run_dispatcher.py"),
    ("analysis", "run_dispatcher_analysis.py"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny GA budget, writes results_smoke/")
    parser.add_argument("--from", dest="start", choices=[s for s, _ in STAGES], default="label",
                        help="first stage to run")
    parser.add_argument("--only", choices=[s for s, _ in STAGES], help="run just this stage")
    parser.add_argument("--redo", action="store_true", help="re-run label/costs even if outputs exist")
    args = parser.parse_args()

    env = dict(os.environ)
    if args.smoke:
        env["PERTINENCE_SMOKE"] = "1"

    import config  # only for output paths, fine to import here
    already_done = {
        "label": os.path.exists(config.TRAIN_GROUND_TRUTH_CSV) and os.path.exists(config.FINAL_VAL_GROUND_TRUTH_CSV),
        "costs": os.path.exists(config.MODEL_COSTS_JSON),
    }

    names = [s for s, _ in STAGES]
    selected = [args.only] if args.only else names[names.index(args.start):]
    for stage, script in STAGES:
        if stage not in selected:
            continue
        if already_done.get(stage) and not args.redo and not args.only:
            print(f"\n=== {stage}: outputs exist, skipping (--redo to force) ===")
            continue
        print(f"\n=== {stage}: python {script} ===", flush=True)
        start = time.time()
        result = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE, env=env)
        if result.returncode != 0:
            sys.exit(f"\nStage '{stage}' failed (exit {result.returncode}). "
                     f"Fix it, then resume with: python run_all.py --from {stage}"
                     + (" --smoke" if args.smoke else ""))
        print(f"=== {stage}: done in {time.time() - start:.0f}s ===")

    print(f"\nAll done. Results: {os.path.join(HERE, 'results_smoke' if args.smoke else 'results')}")


if __name__ == "__main__":
    main()
