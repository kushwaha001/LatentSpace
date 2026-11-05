"""Push adapters, logs and datasets to the Hugging Face Hub (only with ``--push``).

Repo ids come from the stage config (``hub.repos``), defaulting to the ``hub.namespace``
user (``kanishk001``). The token is read from ``HF_TOKEN``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def repo_id(cfg: dict, key: str) -> str:
    hub = cfg.get("hub", {})
    rid = (hub.get("repos") or {}).get(key)
    if not rid:
        raise KeyError(f"hub.repos.{key} is not set in {cfg.get('_config_path')}")
    return rid if "/" in rid else f"{hub.get('namespace', 'kanishk001')}/{rid}"


def push_folder(cfg: dict, key: str, folder: Path, *, repo_type: str = "model",
                path_in_repo: str | None = None, message: str | None = None,
                allow_patterns: list[str] | None = None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("--push needs HF_TOKEN in the environment")
    rid = repo_id(cfg, key)
    api = HfApi(token=token)
    api.create_repo(rid, repo_type=repo_type, private=cfg.get("hub", {}).get("private", False),
                    exist_ok=True)
    info = api.upload_folder(
        repo_id=rid, repo_type=repo_type, folder_path=str(folder),
        path_in_repo=path_in_repo, commit_message=message or f"{cfg.get('stage')} upload",
        allow_patterns=allow_patterns,
        ignore_patterns=["checkpoint-*/**", "*.pt", "optimizer*", "rng_state*"],
    )
    return {"repo_id": rid, "repo_type": repo_type, "path_in_repo": path_in_repo,
            "commit": getattr(info, "oid", None), "url": getattr(info, "commit_url", str(info))}


def push_file(cfg: dict, key: str, file: Path, *, repo_type: str = "dataset",
              path_in_repo: str | None = None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("--push needs HF_TOKEN in the environment")
    rid = repo_id(cfg, key)
    api = HfApi(token=token)
    api.create_repo(rid, repo_type=repo_type, private=cfg.get("hub", {}).get("private", False),
                    exist_ok=True)
    dest = path_in_repo or file.name
    info = api.upload_file(path_or_fileobj=str(file), path_in_repo=dest, repo_id=rid,
                           repo_type=repo_type, commit_message=f"{cfg.get('stage')}: {dest}")
    return {"repo_id": rid, "repo_type": repo_type, "path_in_repo": dest,
            "url": getattr(info, "commit_url", str(info))}
