"""Upload the eval set and the experiment logs to the Hugging Face Hub (hackathon rule 4:
host all datasets and logs on Hugging Face).

    huggingface-cli login            # a token with write access
    python scripts/push_eval_to_hub.py --dataset <user>/transit-agent-eval \\
                                       --logs <user>/latentspace-logs

- ``--dataset``: data/eval/ (eval_set.jsonl, the three split files, the dataset card,
  DATACARD.md, review_checklist.csv, base_tasks.yaml) and data/backend/ (the mock backend).
  The dataset card's YAML header (data/eval/HF_README.md, uploaded as README.md) defines the
  train / dev / test splits for ``datasets.load_dataset``.
- ``--logs``: results/ (every experiment's JSON, trajectories and score logs).

Both repos are created public unless ``--private``. The hackathon requires public artifacts by
the deadline. Nothing is uploaded with ``--dry-run``; the files that would go up are listed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def plan(dataset: bool) -> list[tuple[Path, str]]:
    if not dataset:
        return [(p, p.relative_to(ROOT / "results").as_posix())
                for p in sorted((ROOT / "results").glob("*")) if p.is_file()]
    files = [(ROOT / "data" / "eval" / "HF_README.md", "README.md")]
    for name in ("eval_set.jsonl", "train.jsonl", "dev.jsonl", "test.jsonl", "DATACARD.md",
                 "review_checklist.csv", "base_tasks.yaml"):
        files.append((ROOT / "data" / "eval" / name, f"data/{name}"))
    for p in sorted((ROOT / "data" / "backend").glob("*")):
        if p.is_file():
            files.append((p, f"backend/{p.name}"))
    return files


def push(repo_id: str, files: list[tuple[Path, str]], private: bool, message: str) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for src, dst in files]
    api.create_commit(repo_id, operations=ops, repo_type="dataset", commit_message=message)
    print(f"https://huggingface.co/datasets/{repo_id}  ({len(ops)} files)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", help="HF dataset repo for the eval set, e.g. user/transit-agent-eval")
    p.add_argument("--logs", help="HF dataset repo for results/, e.g. user/latentspace-logs")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not (a.dataset or a.logs):
        p.error("give --dataset and/or --logs")
    for repo, is_dataset in ((a.dataset, True), (a.logs, False)):
        if not repo:
            continue
        files = plan(is_dataset)
        missing = [str(s) for s, _ in files if not s.exists()]
        if missing:
            raise SystemExit(f"missing files: {missing}")
        if a.dry_run:
            print(repo, *[f"  {d}" for _, d in files], sep="\n")
        else:
            push(repo, files, a.private, "Upload eval set" if is_dataset else "Upload logs")


if __name__ == "__main__":
    main()
