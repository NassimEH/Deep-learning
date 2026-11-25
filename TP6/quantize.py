import argparse
import copy
import functools
import time

import torch
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from common import append_result, benchmark, build_model, make_dataset, median, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="")
    p.add_argument("--ckpt", default="best.pth")
    p.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--calib", type=int, default=200)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--variants", default="fp32-eager,fp32,bf16,int8")
    return p.parse_args()


def spread(n, total):
    return torch.linspace(0, total - 1, min(n, total)).long().tolist()


def weights_mb(module):
    if isinstance(module, torch.fx.GraphModule):
        names = {n.target for n in module.graph.nodes if n.op == "get_attr"}
        tensors = [functools.reduce(getattr, name.split("."), module) for name in names]
        tensors = [t for t in tensors if isinstance(t, torch.Tensor)]
    else:
        tensors = list(module.state_dict().values())
    return sum(t.numel() * t.element_size() for t in tensors) / 1e6


def predict(fn, dataset, idx, dtype=torch.float32):
    return torch.cat([fn(dataset[i][0].unsqueeze(0).to(dtype)).float() for i in idx])


def quantize_int8(model, example, calib_set, n_calib):
    gm = export(model, example, strict=True).module()
    quantizer = xiq.X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
    prepared = prepare_pt2e(gm, quantizer)
    for i in spread(n_calib, len(calib_set)):
        prepared(calib_set[i][0].unsqueeze(0))
    return convert_pt2e(prepared)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    seed_everything(0)
    model = build_model(args.arch)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    val_set = make_dataset("small", "val")
    calib_set = make_dataset("small", "train", augment=False)
    example = (torch.randn(1, 3, 224, 224),)
    idx = spread(args.n_val, len(val_set))
    labels = torch.tensor([val_set.samples[i][1] for i in idx])

    with torch.no_grad():
        ref = predict(model, val_set, idx)
        variants = {}
        wanted = args.variants.split(",")
        if "fp32-eager" in wanted:
            variants["fp32-eager"] = (model, model, torch.float32)
        if "fp32" in wanted:
            variants["fp32"] = (torch.compile(model), model, torch.float32)
        if "bf16" in wanted:
            m16 = copy.deepcopy(model).to(torch.bfloat16)
            variants["bf16"] = (torch.compile(m16), m16, torch.bfloat16)
        if "int8" in wanted:
            q = quantize_int8(model, example, calib_set, args.calib)
            variants["int8"] = (torch.compile(q), q, torch.float32)

        for name, (fn, module, dtype) in variants.items():
            x = example[0].to(dtype)
            size = weights_mb(module)
            t0 = time.perf_counter()
            fn(x)
            first_s = time.perf_counter() - t0
            lat = median(benchmark(lambda: fn(x), "cpu", n_warmup=10, n_iter=50))
            logits = predict(fn, val_set, idx, dtype)
            acc = (logits.argmax(1) == labels).float().mean().item()
            agree = (logits.argmax(1) == ref.argmax(1)).float().mean().item()
            gap = (logits - ref).abs().mean().item()
            row = {"tag": args.tag, "variante": name, "threads": args.threads,
                   "calibration": args.calib if name == "int8" else "",
                   "taille_mo": round(size, 1), "latence_ms": round(1e3 * lat, 2),
                   "premier_appel_s": round(first_s, 1), "top1_val": round(acc, 4),
                   "accord_fp32": round(agree, 4), "ecart_logits": round(gap, 4),
                   "n_val": len(idx)}
            print(" | ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            append_result(row, "results_quant.csv")


if __name__ == "__main__":
    main()
