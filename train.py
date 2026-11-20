"""Train on fixed, disjoint CIFAR-10 train/validation/test subsets."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from core import ColorUNet, ROOT, as_lab

SOURCE = "https://huggingface.co/datasets/uoft-cs/cifar10/tree/0b27149/plain_text"
def digest(path):
    with path.open("rb") as handle:
        result = sha256()
        for part in iter(lambda: handle.read(2**20), b""):
            result.update(part)
    return result.hexdigest()

def read_dataset(path):
    data = pq.read_table(path, columns=["img"])["img"]
    return data

def decode(data, indices):
    def image(i):
        with Image.open(BytesIO(data[int(i)].as_py()["bytes"])) as rgb:
            return np.array(rgb.convert("RGB"), dtype=np.uint8)
    return np.stack([image(i) for i in indices])

def make_tensor(images):
    output = np.stack([as_lab(im) for im in images]).transpose(0,3,1,2)
    return TensorDataset(torch.from_numpy(output[:,:1]/100),torch.from_numpy(output[:,1:]/128))

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    model_sum = base_sum = count = 0
    for light, ab in loader:
        pred = model(light)
        model_sum += (pred-ab).abs().sum().item()
        base_sum += ab.abs().sum().item()
        count += ab.numel()
    return round(model_sum/count*128,3),round(base_sum/count*128,3)

def train(args):
    torch.manual_seed(17)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(17)
    start = time.perf_counter()
    all_train, all_test = (read_dataset(args.train_parquet), read_dataset(args.test_parquet))
    idx = rng.permutation(len(all_train))
    test_idx = rng.choice(len(all_test), args.test, replace=False)
    train_loader = DataLoader(make_tensor(decode(all_train,idx[:args.train])), batch_size=64, shuffle=True)
    val_loader = DataLoader(make_tensor(decode(all_train,idx[args.train:args.train+args.val])), batch_size=128)
    test_loader = DataLoader(make_tensor(decode(all_test,test_idx)), batch_size=128)
    model = ColorUNet().to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.MSELoss()
    best = float("inf")
    history = []
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        for light, ab in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(light.contiguous(memory_format=torch.channels_last)),ab)
            loss.backward()
            opt.step()
        v, _ = evaluate(model,val_loader)
        print(f"epoch {epoch}: validation ab MAE {v:.3f}",flush=True)
        history.append(v)
        if v < best:
            best = v
            torch.save({"state":model.state_dict(),"size":32},args.weights)
    model.load_state_dict(torch.load(args.weights,weights_only=True,map_location="cpu")["state"])
    score, neutral = evaluate(model,test_loader)
    report = {
        "architecture":"one-level LAB U-Net", "device":"CPU", "dataset":"CIFAR-10",
        "train_sha256":digest(args.train_parquet),"test_sha256":digest(args.test_parquet),
        "seed":17,"size":32,"train_images":args.train,"validation_images":args.val,
        "test_images":args.test,"epochs":args.epochs,"validation_ab_mae":history,
        "selected_epoch":int(np.argmin(history))+1,"test_ab_mae":score,
        "neutral_chroma_test_ab_mae":neutral,"elapsed_seconds":round(time.perf_counter()-start,1)
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_parquet",type=Path)
    parser.add_argument("test_parquet",type=Path)
    parser.add_argument("--weights",type=Path,default=ROOT/".work/retrained.pt")
    parser.add_argument("--report",type=Path,default=ROOT/".work/retrain-evaluation.json")
    parser.add_argument("--train",type=int,default=3072)
    parser.add_argument("--val",type=int,default=512)
    parser.add_argument("--test",type=int,default=1024)
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if min(args.train,args.val,args.test,args.epochs,args.threads)<1 or args.train+args.val>50000 or args.test>10000:
        parser.error("Invalid split size, epoch count, or threads.")
    train(args)
