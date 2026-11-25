import os
import socket
import time

import torch
import torch.distributed as dist


def main():
    use_cuda = torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device) if use_cuda else "CPU"
    print(f"[rang {rank}/{world}] {socket.gethostname()} : {name}", flush=True)

    t = torch.tensor([float(rank)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    assert t.item() == world * (world - 1) / 2, "la communication est incorrecte"

    buf = torch.ones(25_557_032, device=device)
    for _ in range(3):
        dist.all_reduce(buf)
    if use_cuda:
        torch.cuda.synchronize()
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        dist.all_reduce(buf)
    if use_cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    size = buf.numel() * buf.element_size()
    algbw = size / dt
    busbw = algbw * (2 * (world - 1) / world)
    if rank == 0:
        print(f"all-reduce de {size / 1e6:.0f} Mo sur {world} processus : {dt * 1e3:.1f} ms ; "
              f"débit algorithmique {algbw / 1e9:.2f} Go/s ; débit par lien {busbw / 1e9:.2f} Go/s",
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
