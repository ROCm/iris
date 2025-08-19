import sys
from typing import List, Dict, Any, Optional

import torch

def dist_print(message: str, rank: int, is_error: bool = False):
    """Prints a message only from rank 0."""
    if rank == 0:
        if is_error:
            print(f"❌ ERROR: {message}", file=sys.stderr)
        else:
            print(message)

def print_correctness_report(
    rank: int,
    computed: torch.Tensor,
    reference: torch.Tensor,
    error: Optional[Exception] = None
):
    """
    Prints a detailed report from rank 0 and the final status from all ranks.
    """
    # This detailed report prints only from Rank 0
    if rank == 0:
        print(f"\n<<<<<<<<<< Correctness Test Report (Impl: FUSED_FULL) >>>>>>>>>>")
        print(f"--- Detailed Validation on Rank {rank} ---")
        header = f"{'Index':<8} | {'Computed':<15} | {'Reference':<15} | {'Abs. Diff':<15}"
        print("--- Comparison of First 16 Values (Head 0) ---")
        print(header)
        print("-" * len(header))
        
        comp_slice = computed[0, 0, :16].cpu().float()
        ref_slice = reference[0, 0, :16].cpu().float()
        diff_slice = torch.abs(comp_slice - ref_slice)
        
        for i in range(len(comp_slice)):
            print(f"{i:<8} | {comp_slice[i]:<15.6f} | {ref_slice[i]:<15.6f} | {diff_slice[i]:<15.6f}")
        print("-" * len(header))

    # This final status prints from ALL ranks
    if error:
        print(f"❌ TEST FAILED for Rank {rank}:\n{error}")
    else:
        max_diff = torch.max(torch.abs(computed - reference))
        print(f"✅ TEST PASSED for Rank {rank}. Max absolute difference: {max_diff:.6f}")

class PerfLogger:
    """A utility clas rank-aware logging for the performance test."""
    def __init__(self, rank: int):
        self.rank = rank

    def log(self, message: str, is_error: bool = False):
        dist_print(message, self.rank, is_error)
    
    def log_header(self, title: str, char: str = "=", length: int = 80):
        self.log(f"\n{title.center(length, ' ')}\n{char * length}")

    def log_perf_summary(self, results: List[Dict[str, Any]], num_gpus: int):
        if self.rank != 0 or not results:
            return
            
        impl_name = "FUSED_FULL"
        self.log_header(f"Final Performance Summary (Implementation: {impl_name} | Num GPUs: {num_gpus})", char="-")
        
        header = f"{'KV Len/GPU':<15} | {impl_name + ' (ms)':<18}"
        self.log(header)
        self.log("-" * len(header))
        
        for r in results:
            row = f"{r['kv_len']:<15} | {r.get(impl_name, 'N/A'):<18.3f}"
            self.log(row)
        self.log("-" * len(header))