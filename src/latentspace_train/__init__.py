"""LatentSpace training pipeline: baseline -> teacher_collect -> distill -> grpo -> evaluate.

One CLI, one YAML config per stage (``configs/``):

    python -m latentspace_train baseline        --config configs/10_baseline.yaml
    python -m latentspace_train teacher_collect --config configs/11_teacher_collect.yaml
    python -m latentspace_train distill         --config configs/12_distill.yaml
    python -m latentspace_train grpo            --config configs/13_grpo.yaml --run run_A
    python -m latentspace_train grpo            --config configs/13_grpo.yaml --run run_B
    python -m latentspace_train evaluate        --config configs/14_evaluate.yaml

See README.md in this directory.
"""

STAGES = ("baseline", "teacher_collect", "distill", "grpo", "evaluate")

__all__ = ["STAGES"]
