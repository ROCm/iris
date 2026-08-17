#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Model architecture configurations for LLM benchmarking.

This module provides a centralized registry of model parameters (hidden_size,
attention_output_size, intermediate_size) for generating benchmark dimension configurations across
different operations and batch sizes.

Usage:
    >>> from model_configs import MODELS, compute_dimensions, OperationType
    >>> llama3_8b = MODELS["llama3_8b"]
    >>> dims = compute_dimensions(llama3_8b, OperationType.ATTN_OUT, batch_size=16384, tp_degree=8)
    >>> print(f"{dims.label}: M={dims.m}, N={dims.n}, K={dims.k}")
    llama3_8b_attn_out_16k: M=16384, N=4096, K=4096

    >>> # Compute local dimensions after TP sharding
    >>> print(f"K_local (sharded): {dims.k_local}")
    K_local (sharded): 512

Adding a new model:
    1. Add entry to MODELS dict with hidden_size, attention_output_size, and intermediate_size
    2. Optionally add full_name and notes for documentation
    3. Run model_sweep.py to validate TP sharding divides evenly

    Example:
        MODELS["new_model"] = ModelConfig(
            name="new_model",
            hidden_size=4096,
            attention_output_size=4096,
            intermediate_size=16384,
            full_name="New Model 100B",
            notes="Architecture details or source"
        )
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture parameters for LLM benchmarking.

    These parameters define the core tensor dimensions used in transformer layers.
    Only the minimal set needed to compute matmul shapes for TP benchmarking.

    Attributes:
        name: Canonical model identifier (e.g., "llama3_8b")
        hidden_size: d_model - token embedding dimension (attention input/output)
        attention_output_size: Concatenated attention-head output width before the output projection
        num_attention_heads: Number of query/attention heads
        head_dim: Conventional Q/K/V head width when the architecture uses one common width
        value_head_dim: Value-head width when it differs from the Q/K head width
        intermediate_size: FFN intermediate dimension (after up-projection, before down-projection)
        expert_intermediate_size: Per-expert FFN width for MoE layers, if present
        num_routed_experts: Total number of routed experts in each MoE layer
        num_experts_per_token: Number of routed experts selected for each token
        num_shared_experts: Number of experts that process every token
        num_dense_layers: Number of transformer layers that use the dense FFN
        active_moe_intermediate_size: Calculated aggregate width of active routed and shared experts
        full_name: Human-readable model name (e.g., "Llama 3 8B")
        notes: Source, estimation details, or architecture notes
    """
    name: str
    hidden_size: int
    attention_output_size: int
    intermediate_size: int
    num_attention_heads: int | None = None
    head_dim: int | None = None
    value_head_dim: int | None = None
    expert_intermediate_size: int | None = None
    num_routed_experts: int = 0
    num_experts_per_token: int = 0
    num_shared_experts: int = 0
    num_dense_layers: int | None = None
    full_name: str = ""
    notes: str = ""

    @property
    def active_moe_intermediate_size(self) -> int | None:
        """Aggregate intermediate width active for each token in an MoE layer."""
        if self.expert_intermediate_size is None:
            return None
        return self.expert_intermediate_size * (self.num_experts_per_token + self.num_shared_experts)


class OperationType(Enum):
    """LLM matmul operations (layer types in transformer architectures).

    These correspond to different linear layers:
    - ATTN_OUT: Attention output projection
    - MLP_DOWN: MLP down-projection from FFN to hidden
    - EXPERT_MLP_DOWN: Per-expert MoE down-projection from expert FFN to hidden
    - ACTIVE_MOE_MLP_DOWN: Aggregate active routed and shared expert down-projections
    - MLP_UP: MLP up-projection from hidden to FFN
    """
    ATTN_OUT = "attn_out"
    MLP_DOWN = "mlp_down"
    EXPERT_MLP_DOWN = "expert_mlp_down"
    ACTIVE_MOE_MLP_DOWN = "active_moe_mlp_down"
    MLP_UP = "mlp_up"


class SweepOperation(Enum):
    """Communication patterns for tensor-parallel matmul operations.

    These correspond to the benchmark sweep operations that test different
    communication/compute fusion strategies:
    - ALL_GATHER_MATMUL: All-gather input, then matmul (K-sharding)
    - MATMUL_ALL_GATHER: Matmul, then all-gather output (N-sharding post-compute)
    - MATMUL_ALL_REDUCE: Matmul, then all-reduce output (N-sharding column-parallel)
    """
    ALL_GATHER_MATMUL = "all_gather_matmul"
    MATMUL_ALL_GATHER = "matmul_all_gather"
    MATMUL_ALL_REDUCE = "matmul_all_reduce"


@dataclass
class DimensionSpec:
    """Computed dimension specification for a benchmark configuration.

    Contains both global dimensions (M, N, K) and local per-rank dimensions
    after TP sharding. Global dimensions are used for:
    - Heap size calculation
    - Result labeling
    - Configuration matching

    Local dimensions are used for:
    - Actual benchmark kernel invocation
    - Memory allocation per rank

    Attributes:
        m: Global M dimension (batch size * sequence length)
        n: Global N dimension (hidden_size or intermediate_size)
        k: Global K dimension (hidden_size or intermediate_size)
        label: Benchmark label (e.g., "llama3_8b_attn_out_16k")
        m_local: Per-rank M (equals m unless M-sharding is used)
        n_local: Per-rank N (equals n unless N-sharding is used)
        k_local: Per-rank K (equals k unless K-sharding is used)
        operation: Operation type string
        model_name: Source model name
        batch_size: Number of tokens (M dimension before TP)
    """
    m: int
    n: int
    k: int
    label: str
    m_local: int
    n_local: int
    k_local: int
    operation: str
    model_name: str
    batch_size: int


# Model registry
MODELS: Dict[str, ModelConfig] = {
    "deepseek_v3": ModelConfig(
        name="deepseek_v3",
        hidden_size=7168,
        attention_output_size=16384,
        intermediate_size=18432,
        num_attention_heads=128,
        value_head_dim=128,
        expert_intermediate_size=2048,
        num_routed_experts=256,
        num_experts_per_token=8,
        num_shared_experts=1,
        num_dense_layers=3,
        full_name="DeepSeek-V3",
        notes="671B params; 3 dense layers followed by 58 MoE layers",
    ),
    "deepseek_v4": ModelConfig(
        name="deepseek_v4",
        hidden_size=7168,
        attention_output_size=16384,
        intermediate_size=3072,
        num_attention_heads=128,
        head_dim=512,
        expert_intermediate_size=3072,
        num_routed_experts=384,
        num_experts_per_token=6,
        num_shared_experts=1,
        num_dense_layers=0,
        full_name="DeepSeek-V4 Pro",
        notes="1.6T params, 384-expert MoE",
    ),
    "llama3_8b": ModelConfig(
        name="llama3_8b",
        hidden_size=4096,
        attention_output_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        head_dim=128,
        full_name="Llama 3/3.1 8B",
    ),
    "llama3_70b": ModelConfig(
        name="llama3_70b",
        hidden_size=8192,
        attention_output_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        head_dim=128,
        full_name="Llama 3/3.1 70B",
    ),
    "llama3_405b": ModelConfig(
        name="llama3_405b",
        hidden_size=16384,
        attention_output_size=16384,
        intermediate_size=53248,
        num_attention_heads=128,
        head_dim=128,
        full_name="Llama 3.1 405B",
    ),
    "llama4_scout": ModelConfig(
        name="llama4_scout",
        hidden_size=5120,
        attention_output_size=5120,
        intermediate_size=8192,
        num_attention_heads=40,
        head_dim=128,
        expert_intermediate_size=8192,
        num_routed_experts=16,
        num_experts_per_token=1,
        num_shared_experts=1,
        num_dense_layers=0,
        full_name="Llama 4 Scout",
        notes="17B active params, 16-expert MoE",
    ),
    "gpt_oss_120b": ModelConfig(
        name="gpt_oss_120b",
        hidden_size=2880,
        attention_output_size=4096,
        intermediate_size=2880,
        num_attention_heads=64,
        head_dim=64,
        expert_intermediate_size=2880,
        num_routed_experts=128,
        num_experts_per_token=4,
        num_shared_experts=0,
        num_dense_layers=0,
        full_name="GPT-OSS 120B",
        notes="Open-weight GPT model, 128-expert MoE",
    ),
}


def _validate_tp_sharding(dim: int, tp_degree: int, dim_name: str, operation: str):
    """Validate that TP degree evenly divides the sharded dimension.

    Args:
        dim: Dimension size to check
        tp_degree: Tensor parallelism degree (world size)
        dim_name: Human-readable dimension name for error messages
        operation: Operation type for context in error messages

    Raises:
        ValueError: If TP degree doesn't divide dimension evenly
    """
    if dim % tp_degree != 0:
        raise ValueError(
            f"TP degree {tp_degree} does not evenly divide {dim_name}={dim} "
            f"for operation '{operation}'. Remainder: {dim % tp_degree}. "
            f"Adjust model config or use a TP degree that divides {dim_name}."
        )


def compute_dimensions(
    model: ModelConfig,
    operation_type: OperationType,
    sweep_operation: SweepOperation,
    batch_size: int,
    tp_degree: int = 8,
) -> DimensionSpec:
    """Compute benchmark dimensions based on layer type and communication pattern.

    Dimension calculation depends on BOTH the layer type (operation_type) and the
    communication pattern (sweep_operation):

    ALL_GATHER_MATMUL (gather input along K, then matmul):
        - ATTN_OUT: M=batch, N=hidden, K=attention output (K-sharded)
        - MLP_DOWN: M=batch, N=hidden, K=intermediate (K-sharded)
        - EXPERT_MLP_DOWN: M=batch, N=hidden, K=expert intermediate (K-sharded)
        - ACTIVE_MOE_MLP_DOWN: M=batch, N=hidden, K=aggregate active MoE width (K-sharded)

    MATMUL_ALL_REDUCE (matmul with sharded weights, then all-reduce):
        - MLP_DOWN: M=batch, N=intermediate, K=hidden (N-sharded)
        - EXPERT_MLP_DOWN: M=batch, N=expert intermediate, K=hidden (N-sharded)
        - ACTIVE_MOE_MLP_DOWN: M=batch, N=aggregate active MoE width, K=hidden (N-sharded)
        - ATTN_OUT: M=batch, N=hidden, K=attention output (K-sharded)

    MATMUL_ALL_GATHER (matmul with sharded weights, then all-gather output):
        - MLP_DOWN: M=batch, N=intermediate, K=hidden (N-sharded, column-parallel)
        - EXPERT_MLP_DOWN: M=batch, N=expert intermediate, K=hidden (N-sharded)
        - ACTIVE_MOE_MLP_DOWN: M=batch, N=aggregate active MoE width, K=hidden (N-sharded)

    Args:
        model: Model configuration with hidden_size, attention_output_size, and intermediate_size
        operation_type: Layer type (ATTN_OUT, MLP_DOWN, EXPERT_MLP_DOWN, ACTIVE_MOE_MLP_DOWN, MLP_UP)
        sweep_operation: Communication pattern (ALL_GATHER_MATMUL, MATMUL_ALL_REDUCE, etc.)
        batch_size: Number of tokens (M dimension before TP)
        tp_degree: Tensor parallelism degree (must divide sharded dimensions evenly)

    Returns:
        DimensionSpec with both global and local (per-rank) dimensions

    Raises:
        ValueError: If TP degree doesn't divide sharded dimensions evenly

    Examples:
        >>> llama3_8b = MODELS["llama3_8b"]
        >>> # all_gather_matmul: gather along K
        >>> dims = compute_dimensions(llama3_8b, OperationType.MLP_DOWN,
        ...                          SweepOperation.ALL_GATHER_MATMUL, 16384, 8)
        >>> dims.m, dims.n, dims.k
        (16384, 4096, 14336)  # N=hidden, K=intermediate

        >>> # matmul_all_reduce: N-sharded column-parallel
        >>> dims = compute_dimensions(llama3_8b, OperationType.MLP_DOWN,
        ...                          SweepOperation.MATMUL_ALL_REDUCE, 16384, 8)
        >>> dims.m, dims.n, dims.k
        (16384, 14336, 4096)  # N=intermediate, K=hidden (N and K swapped!)
    """
    h = model.hidden_size
    attn_out = model.attention_output_size
    if operation_type in (OperationType.EXPERT_MLP_DOWN, OperationType.ACTIVE_MOE_MLP_DOWN):
        if model.expert_intermediate_size is None:
            raise ValueError(f"Model '{model.name}' does not define an expert intermediate size")
        if operation_type == OperationType.ACTIVE_MOE_MLP_DOWN:
            ffn = model.active_moe_intermediate_size
            ffn_name = "active_moe_intermediate_size"
        else:
            ffn = model.expert_intermediate_size
            ffn_name = "expert_intermediate_size"
    else:
        ffn = model.intermediate_size
        ffn_name = "intermediate_size"

    # Compute dimensions based on (operation_type, sweep_operation) combination
    if sweep_operation == SweepOperation.ALL_GATHER_MATMUL:
        # All-gather input, then matmul (K-sharding)
        if operation_type == OperationType.ATTN_OUT:
            m, n, k = batch_size, h, attn_out
            k_local = k // tp_degree
            m_local, n_local = m, n
            _validate_tp_sharding(k, tp_degree, "K (attention_output_size)", operation_type.value)
        elif operation_type in (
            OperationType.MLP_DOWN,
            OperationType.EXPERT_MLP_DOWN,
            OperationType.ACTIVE_MOE_MLP_DOWN,
        ):
            m, n, k = batch_size, h, ffn
            k_local = k // tp_degree
            m_local, n_local = m, n
            _validate_tp_sharding(k, tp_degree, f"K ({ffn_name})", operation_type.value)
        else:
            raise ValueError(f"Unsupported operation_type {operation_type} for {sweep_operation}")

    elif sweep_operation == SweepOperation.MATMUL_ALL_REDUCE:
        # Matmul with row-parallel weights, then all-reduce (K-sharding)
        # Each rank has partial input and weight slice, computes partial sum, then all-reduce
        if operation_type in (
            OperationType.MLP_DOWN,
            OperationType.EXPERT_MLP_DOWN,
            OperationType.ACTIVE_MOE_MLP_DOWN,
        ):
            # [batch, intermediate/tp] @ [intermediate/tp, hidden] → all-reduce SUM
            # Row-parallel: K dimension (intermediate) is sharded
            m, n, k = batch_size, h, ffn
            k_local = k // tp_degree
            m_local, n_local = m, n
            _validate_tp_sharding(k, tp_degree, f"K ({ffn_name})", operation_type.value)
        elif operation_type == OperationType.ATTN_OUT:
            # [batch, attention_output/tp] @ [attention_output/tp, hidden] → all-reduce SUM
            # Row-parallel: K dimension (attention output) is sharded
            m, n, k = batch_size, h, attn_out
            k_local = k // tp_degree
            m_local, n_local = m, n
            _validate_tp_sharding(k, tp_degree, "K (attention_output_size)", operation_type.value)
        else:
            raise ValueError(f"Unsupported operation_type {operation_type} for {sweep_operation}")

    elif sweep_operation == SweepOperation.MATMUL_ALL_GATHER:
        # Matmul with column-parallel weights, then all-gather output (N-sharding)
        if operation_type in (
            OperationType.MLP_DOWN,
            OperationType.EXPERT_MLP_DOWN,
            OperationType.ACTIVE_MOE_MLP_DOWN,
        ):
            # Column-parallel: [batch, hidden] @ [hidden, intermediate/tp] → all-gather
            m, n, k = batch_size, ffn, h
            n_local = n // tp_degree
            m_local, k_local = m, k
            _validate_tp_sharding(n, tp_degree, f"N ({ffn_name})", operation_type.value)
        else:
            raise ValueError(f"Unsupported operation_type {operation_type} for {sweep_operation}")

    else:
        raise ValueError(f"Unknown sweep operation: {sweep_operation}")

    # Format batch size for label (16384 → "16k", 32768 → "32k", 1 → "1", 32 → "32")
    if batch_size >= 1024 and batch_size % 1024 == 0:
        batch_label = f"{batch_size // 1024}k"
    else:
        batch_label = str(batch_size)

    label = f"{model.name}_{operation_type.value}_{batch_label}"

    return DimensionSpec(
        m=m,
        n=n,
        k=k,
        m_local=m_local,
        n_local=n_local,
        k_local=k_local,
        label=label,
        operation=operation_type.value,
        model_name=model.name,
        batch_size=batch_size,
    )
