"""Test gfx1250 async copy kernel with 2 simulated ranks on 1 GPU."""
import os

os.environ["IRIS_SIMULATION"] = "1"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import torch
import torch.multiprocessing as mp


def run_rank(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist

    torch.cuda.set_device(0)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    import iris.experimental.iris_gluon as ig
    from iris.ccl.config import Config

    ctx = ig.iris(heap_size=1 << 28)
    r = ctx.get_rank()
    ws = ctx.get_num_ranks()
    print(f"[rank {r}] initialized, world_size={ws}")

    M, N = 64, 64
    t_in = ctx.zeros((M, N), dtype=torch.float16)
    t_in.fill_(float(r + 1))
    t_out = ctx.zeros((ws * M, N), dtype=torch.float16)

    config = Config(use_gluon=True, block_size_m=32, block_size_n=64, comm_sms=4)

    ctx.barrier()
    ctx.ccl.all_gather(t_out, t_in, config=config)
    torch.cuda.synchronize()

    # Validate
    passed = True
    for i in range(ws):
        expected = float(i + 1)
        chunk = t_out[i * M : (i + 1) * M]
        if not torch.allclose(chunk, torch.full_like(chunk, expected), atol=0.5):
            print(f"[rank {r}] FAIL: chunk {i} got {chunk[0,0].item():.1f}, expected {expected:.1f}")
            passed = False

    if passed:
        print(f"[rank {r}] PASS")
    else:
        print(f"[rank {r}] FAIL")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    mp.set_start_method("spawn")
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run_rank, args=(rank, world_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    exit_codes = [p.exitcode for p in processes]
    print(f"Exit codes: {exit_codes}")
    if all(c == 0 for c in exit_codes):
        print("ALL PASSED")
    else:
        print("SOME FAILED")
