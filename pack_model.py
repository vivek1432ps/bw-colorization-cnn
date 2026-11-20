"""Pack a trained checkpoint into portable FP16 NumPy shards."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import torch

from core import WEIGHTS

def pack(checkpoint, dest=WEIGHTS):
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    buffer = BytesIO()
    np.savez_compressed(buffer, **{name: t.numpy().astype(np.float16)
                                for name,t in original["state"].items()})
    payload = buffer.getvalue()
    parts=[]
    chunk=192*1024
    for i,offset in enumerate(range(0,len(payload),chunk),1):
        filename = f"unet-fp16-{i:02}.bin"
        (dest.parent/filename).write_bytes(payload[offset:offset+chunk])
        parts.append(filename)
    manifest={"format":"numpy-state-dict-fp16","size":int(original["size"]),
              "sha256":sha256(payload).hexdigest(),"parts":parts}
    dest.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    args=parser.parse_args()
    pack(args.checkpoint)
