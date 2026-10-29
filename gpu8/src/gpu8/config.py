"""Frozen protocol / model / hardware configuration.

Every knob the research plan pins down lives here.  The configuration is
written to ``configs/frozen_config.json`` before the formal runs start and
hashed into every cache key and run record, so a silent mid-experiment change
is detectable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from . import paths
from .hashing import atomic_write_json, read_json, sha256_obj


# --------------------------------------------------------------------------
# Data construction (plan section 5)
# --------------------------------------------------------------------------
@dataclass
class DataConfig:
    # CTD direct evidence semantics: the primary task studies marker/mechanism.
    primary_evidence: tuple[str, ...] = ("marker/mechanism",)
    secondary_evidence: tuple[str, ...] = ("therapeutic",)
    include_inferred: bool = False
    min_inference_score: float = 50.0
    ppi_score_threshold: int = 700
    ppi_secondary_threshold: int = 700          # second (functional) network
    gene_background: str = "protein_coding"      # HGNC locus_group filter
    keep_legacy_gene_space: bool = True          # bridge to the 8,676-gene space
    hpo_negative_qualifier: str = "NOT"          # never treated as positive


# --------------------------------------------------------------------------
# Splits and query isolation (plan section 6)
# --------------------------------------------------------------------------
@dataclass
class ProtocolConfig:
    dev_family_fraction: float = 0.15            # held-out development families
    n_outer_folds: int = 5
    inner_blocks: int = 3
    family_split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    external_holdout: bool = True
    split_seed: int = 20260922
    support_doses: tuple[int, ...] = (0, 1, 3, 5)
    support_order_draws: int = 10
    min_positives_for_dose: int = 6              # need S_max(5) + >=1 query
    train_seeds: tuple[int, ...] = (42, 52, 62)
    ablation_folds: tuple[int, ...] = (0, 2, 4)
    ablation_seeds: tuple[int, ...] = (42, 52)


# --------------------------------------------------------------------------
# Candidate generation (plan section 7)
# --------------------------------------------------------------------------
@dataclass
class CandidateConfig:
    n_candidates: int = 500
    stress_sizes: tuple[int, ...] = (100, 500, 1000)
    phenotype_neighbours: int = 20
    hierarchy_max_distance: int = 2
    # fusion weights frozen on the development families by tools/tune_retriever.py
    fusion_weight_phenotype: float = 1.0
    fusion_weight_hierarchy: float = 1.0
    fusion_weight_rwr: float = 0.25
    n_anchors: int = 64
    anchor_uniform_fraction: float = 0.5
    anchor_seed: int = 20260922


# --------------------------------------------------------------------------
# Sparse precomputation and neural inputs (plan section 8)
# --------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    view_dim: int = 128
    n_views: int = 4                             # V0..V3 -> 512 per node
    hpo_basis_dim: int = 128
    propagation_steps: int = 2                   # P Z and P^2 Z
    rwr_restart: float = 0.15
    rwr_iters: int = 32
    channel_log_c: float = 1.0                   # log(1 + c x)
    clip_sigma: float = 5.0
    n_scalar_channels: int = 4
    n_quality_fields: int = 12


# --------------------------------------------------------------------------
# Model and adaptation (plan section 9)
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    node_in_dim: int = 512
    hidden_dim: int = 128
    pair_hidden_dim: int = 128
    adapt_dim: int = 64
    channel_net_hidden: int = 32
    control_net_hidden: int = 16
    lambda_min: float = 0.1
    lambda_max: float = 2.0
    rho_max: float = 2.0
    ridge_b_clip: float = 5.0


# --------------------------------------------------------------------------
# Training (plan section 10); 16 GB profile
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    tasks_per_microbatch: int = 8                # 12/16 GB profile
    accumulate_to_tasks: int = 16
    score_batch: int = 8192
    max_epochs: int = 80
    patience: int = 10
    eta_zero_shot: float = 0.25                  # weight of the base-score loss
    task_mix: tuple[float, float, float, float] = (0.5, 0.3, 0.1, 0.1)
    max_query_pos_per_task: int = 16
    max_query_anchor_per_task: int = 32
    missing_view_probs: tuple[float, float, float] = (0.6, 0.25, 0.15)
    amp: bool = True                             # linear layers only
    select_metric: str = "recall10_end"
    tie_break_metric: str = "ndcg10_end"


# --------------------------------------------------------------------------
# Evaluation and statistics (plan sections 7.3 and 16)
# --------------------------------------------------------------------------
@dataclass
class EvalConfig:
    ks: tuple[int, ...] = (10, 20, 50)
    primary_k: int = 10
    bootstrap_draws: int = 10000
    bootstrap_unit: str = "family"
    permutation_draws: int = 10000
    stat_seed: int = 20260922
    multiplicity: str = "holm"
    primary_comparisons: tuple[str, ...] = (
        "b9_ours__vs__b4_lightgbm",
        "b9_ours__vs__b7_kdgene",
        "b9_ours__vs__b8_meta_ridge",
    )


@dataclass
class HardwareConfig:
    profile: str = "16gb"
    device: str = "cuda"
    cpu_threads: int = 4
    target_gpu_reserved_gib: float = 12.0
    target_rss_gib: float = 16.0


@dataclass
class Gpu8Config:
    data: DataConfig = field(default_factory=DataConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    protocol_id: str = "gpu8_v1"

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return sha256_obj(self.to_dict())

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else paths.CONFIG_DIR / "frozen_config.json"
        payload = self.to_dict()
        payload["_config_hash"] = self.config_hash
        return atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Gpu8Config":
        path = Path(path) if path else paths.CONFIG_DIR / "frozen_config.json"
        raw = read_json(path)
        if raw is None:
            return cls()
        raw.pop("_config_hash", None)

        def build(dc, blob):
            if blob is None:
                return dc()
            fields = {f for f in dc.__dataclass_fields__}
            kwargs = {}
            for k, v in blob.items():
                if k not in fields:
                    continue
                # dataclass tuples come back from JSON as lists
                if isinstance(v, list):
                    v = tuple(v)
                kwargs[k] = v
            return dc(**kwargs)

        return cls(
            data=build(DataConfig, raw.get("data")),
            protocol=build(ProtocolConfig, raw.get("protocol")),
            candidates=build(CandidateConfig, raw.get("candidates")),
            features=build(FeatureConfig, raw.get("features")),
            model=build(ModelConfig, raw.get("model")),
            train=build(TrainConfig, raw.get("train")),
            evaluation=build(EvalConfig, raw.get("evaluation")),
            hardware=build(HardwareConfig, raw.get("hardware")),
            protocol_id=raw.get("protocol_id", "gpu8_v1"),
        )


def load_config(path: Path | str | None = None) -> Gpu8Config:
    return Gpu8Config.load(path)
