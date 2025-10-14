#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import Tuple, Optional


##Quantize FP16 tensor to FP8
def quantize_fp16_to_fp8(
    input_tensor: torch.Tensor, scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        max_val = input_tensor.abs().max()
        scale = max_val / 448.0  # FP8 E4M3 max
        scale = torch.clamp(scale, min=1e-8)

    scaled = input_tensor / scale
    fp8_max = 448.0
    clamped = torch.clamp(scaled, -fp8_max, fp8_max)
    quantized = clamped.to(torch.float16)  # Placeholder for FP8

    return quantized, scale


def test_post_quantization_allgather():
    M, N = 128, 1024
    world_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    torch.manual_seed(42)

    # Create 8 input tensors
    input_tensors = []
    for i in range(world_size):
        tensor = torch.randn(M, N, device=device, dtype=dtype) * (i + 1)
        input_tensors.append(tensor)

    print(f"Test setup: {M}×{N} tensors, world_size={world_size}")

    # Create RMSNorm layer
    rmsnorm_layer = nn.RMSNorm(N, eps=1e-6, device=device, dtype=dtype)

    # APPROACH 1: All-Reduce → RMSNorm → Quantization (REFERENCE)

    # All-reduce: sum all tensors
    all_reduced = torch.zeros(M, N, device=device, dtype=dtype)
    for tensor in input_tensors:
        all_reduced += tensor

    print(f"All-reduced sum: {all_reduced.sum():.4f}")

    # RMSNorm using PyTorch built-in
    normed_all_reduced = rmsnorm_layer(all_reduced)
    print(f"RMSNorm result sum: {normed_all_reduced.sum():.4f}")

    # Quantization
    quantized_all_reduced, scale_all_reduced = quantize_fp16_to_fp8(normed_all_reduced)
    print(f"Quantization scale: {scale_all_reduced:.6f}")
    print(f"Final quantized result sum: {quantized_all_reduced.sum():.4f}")

    # APPROACH 2: Reduce-Scatter → RMSNorm (partial) → Quantization → All-Gather
    print("\n" + "=" * 50)
    print("APPROACH 2: Reduce-Scatter → RMSNorm (partial) → Quantization → All-Gather")
    print("=" * 50)

    n_per_rank = N // world_size

    # Step 1: Reduce-scatter - each rank computes its portion
    rank0_local_sum = torch.zeros(M, n_per_rank, device=device, dtype=dtype)
    for tensor in input_tensors:
        rank0_local_sum += tensor[:, :n_per_rank]

    print(f"Rank 0 local sum shape: {rank0_local_sum.shape}, sum: {rank0_local_sum.sum():.4f}")

    # Step 2: RMSNorm on PARTIAL tensor
    # This is the key question - can we do RMSNorm on partial results?
    print("\n  ATTEMPTING RMSNorm ON PARTIAL TENSOR...")
    print(" This may not be mathematically correct!")

    # Create a smaller RMSNorm for the partial dimension
    partial_rmsnorm = nn.RMSNorm(n_per_rank, eps=1e-6, device=device, dtype=dtype)

    normed_partial = partial_rmsnorm(rank0_local_sum)
    print(f"Partial RMSNorm result sum: {normed_partial.sum():.4f}")

    # Step 3: Quantization on partial result
    quantized_partial, scale_partial = quantize_fp16_to_fp8(normed_partial)
    print(f"Partial quantization scale: {scale_partial:.6f}")
    print(f"Partial quantized sum: {quantized_partial.sum():.4f}")

    # Step 4: All-Gather - collect quantized pieces from all ranks
    print("\n📡 Simulating All-Gather of quantized pieces...")

    gathered_quantized = torch.zeros(M, N, device=device, dtype=dtype)

    # Simulate gathering from all ranks
    for rank in range(world_size):
        start_idx = rank * n_per_rank
        end_idx = (rank + 1) * n_per_rank

        # Each rank computes its local sum and processes it
        local_sum = torch.zeros(M, n_per_rank, device=device, dtype=dtype)
        for tensor in input_tensors:
            local_sum += tensor[:, start_idx:end_idx]

        # Each rank does its own RMSNorm and quantization
        local_partial_rmsnorm = nn.RMSNorm(n_per_rank, eps=1e-6, device=device, dtype=dtype)
        local_normed = local_partial_rmsnorm(local_sum)
        local_quantized, local_scale = quantize_fp16_to_fp8(local_normed)

        # Put in the gathered result
        gathered_quantized[:, start_idx:end_idx] = local_quantized

        if rank == 0:
            print(f"Rank {rank} scale: {local_scale:.6f}")

    print(f"Gathered quantized sum: {gathered_quantized.sum():.4f}")

    # Compare final quantized results
    print("COMPARISON")
    diff = torch.abs(quantized_all_reduced - gathered_quantized)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Approach 1 quantized sum: {quantized_all_reduced.sum():.6f}")
    print(f"Approach 2 quantized sum: {gathered_quantized.sum():.6f}")
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")

    # Check if results are approximately equal
    tolerance = 1e-3
    if max_diff < tolerance:
        print("✅ SUCCESS: Post-quantization All-Gather works!")
        return True
    else:
        print("❌ FAILURE: Results differ significantly")
        print("❌ RMSNorm on partial tensors is NOT equivalent to full tensor RMSNorm")
        return False


def main():
    # Test the alternative approach
    success = test_post_quantization_allgather()

    if not success:
        print("\n❌ CONCLUSION:")
        print("   You CANNOT do All-Gather after RMSNorm and quantization.")
        print("   RMSNorm must operate on the FULL tensor.")
        print("   The correct pipeline is:")
        print("   Reduce-Scatter → All-Gather → RMSNorm → Quantization")

    else:
        print("\n✅ CONCLUSION:")
        print("   Post-quantization All-Gather works!")
        print("   This would be more efficient for communication.")


if __name__ == "__main__":
    main()
