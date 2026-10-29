"""Method registry: identifier, human-readable role, and support applicability."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    family: str
    uses_support: bool
    requires_support: bool      # not applicable at k = 0
    trainable: bool
    label_zh: str = ""
    family_zh: str = ""
    note: str = ""


METHODS: dict[str, MethodSpec] = {
    "b0_degree_prior": MethodSpec(
        "b0_degree_prior", "Gene degree / topology prior", "reference", label_zh="基因度／拓扑先验", family_zh="参照",
        uses_support=False, requires_support=False, trainable=False,
        note="Query disease does not enter the score; deliberately not called Adamic-Adar."),
    "b1_phenotype_vote": MethodSpec(
        "b1_phenotype_vote", "Phenotype neighbour voting", "reference", label_zh="表型邻居投票", family_zh="参照",
        uses_support=False, requires_support=False, trainable=False,
        note="Tests how much simple disease transfer already explains."),
    "b2_lr": MethodSpec(
        "b2_lr", "Same-feature logistic regression", "engineered", label_zh="同特征逻辑回归", family_zh="工程特征",
        uses_support=False, requires_support=False, trainable=True,
        note="Identical cold features, sampling and scaling; regularisation tuned."),
    "b3_cpu_fusion": MethodSpec(
        "b3_cpu_fusion", "Channel fusion + fixed 32-dim ridge", "engineered", label_zh="通道融合＋固定 32 维岭", family_zh="工程特征",
        uses_support=True, requires_support=False, trainable=False,
        note="Previous CPU design: fixed projection, no learning through the solver."),
    "b4_lightgbm": MethodSpec(
        "b4_lightgbm", "LightGBM ranker", "tree", label_zh="LightGBM 排序器", family_zh="树模型",
        uses_support=False, requires_support=False, trainable=True,
        note="Disease-grouped lambdarank; tuned, never left at defaults."),
    "b5_mlp": MethodSpec(
        "b5_mlp", "Same-capacity MLP", "neural", label_zh="同容量 MLP", family_zh="神经",
        uses_support=True, requires_support=False, trainable=True,
        note="Matched encoder capacity, no adaptation learning; fixed ridge as its fair support variant."),
    "b6_sign": MethodSpec(
        "b6_sign", "SIGN-style precomputed propagation", "graph", label_zh="SIGN 式预计算传播", family_zh="图学习",
        uses_support=False, requires_support=False, trainable=True,
        note="Task adaptation of the propagation/training separation, not the original model."),
    "b7_kdgene": MethodSpec(
        "b7_kdgene", "KDGene (retrained on the shared graph)", "graph", label_zh="KDGene（共同图重训）", family_zh="图学习",
        uses_support=False, requires_support=False, trainable=True,
        note="Original scoring, gate, full-softmax loss and N3; different entity space, stated."),
    "b8_meta_ridge": MethodSpec(
        "b8_meta_ridge", "Plain meta-ridge adaptation", "neural", label_zh="普通元岭适配", family_zh="神经",
        uses_support=True, requires_support=False, trainable=True,
        note="Nearest-neighbour control: same capacity, equal weights, fixed lambda and rho."),
    "b9_ours": MethodSpec(
        "b9_ours", "Evidence-dependent support adaptation", "neural", label_zh="证据依赖支持适配（本文）", family_zh="神经",
        uses_support=True, requires_support=False, trainable=True,
        note="Proposed method: grouped support weights, learned controls, structured missing training."),
    "b10_ppi_rwr": MethodSpec(
        "b10_ppi_rwr", "PPI random walk with restart", "network", label_zh="PPI 重启随机游走", family_zh="网络",
        uses_support=True, requires_support=True, trainable=False,
        note="Needs disease-specific seeds; not applicable at k = 0."),
    "b10b_diamond": MethodSpec(
        "b10b_diamond", "DIAMOnD-like seed-neighbour enrichment", "network", label_zh="DIAMOnD 式种子邻居富集", family_zh="网络",
        uses_support=True, requires_support=True, trainable=False,
        note="Hypergeometric seed connectivity; reported next to RWR."),
    "b11_quantum_walk": MethodSpec(
        "b11_quantum_walk", "Continuous-time quantum walk", "network", label_zh="连续时间量子游走", family_zh="网络",
        uses_support=True, requires_support=True, trainable=False,
        note="Per-seed amplitudes squared then summed; classical numerical implementation."),
}


def get_method(key: str) -> MethodSpec:
    return METHODS[key]


def list_methods(trainable: bool | None = None,
                 requires_support: bool | None = None) -> list[str]:
    out = []
    for k, m in METHODS.items():
        if trainable is not None and m.trainable != trainable:
            continue
        if requires_support is not None and m.requires_support != requires_support:
            continue
        out.append(k)
    return sorted(out)
