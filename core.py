"""One-level skip U-Net and LAB inference for Luma Lab."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "assets" / "model.json"
MAX_UPLOAD = 8 * 1024**2
MAX_PIXELS = 12_000_000

def block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(),
    )

class ColorUNet(nn.Module):
    """One encoder downsample, decoder upsample, and lateral skip connection."""
    def __init__(self):
        super().__init__()
        self.first = block(1, 64)
        self.deep = block(64, 128)
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.final = block(128, 64)
        self.chroma = nn.Conv2d(64, 2, 1)

    def forward(self, lightness):
        skip = self.first(lightness)
        hidden = self.deep(F.max_pool2d(skip, 2))
        return self.chroma(self.final(torch.cat((self.up(hidden), skip), dim=1)))

def as_lab(rgb):
    return cv2.cvtColor(np.asarray(rgb, dtype=np.float32) / 255, cv2.COLOR_RGB2LAB)

def lab_as_rgb(lab):
    rgb = cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32), cv2.COLOR_LAB2RGB)
    return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)

def read_image(payload):
    if not 0 < len(payload) <= MAX_UPLOAD:
        raise ValueError("Choose an image smaller than 8 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as pic:
                if pic.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Use PNG, JPEG, or WebP.")
                if not 0 < pic.width * pic.height <= MAX_PIXELS:
                    raise ValueError("Use an image of at most 12 megapixels.")
                if getattr(pic, "n_frames", 1) != 1:
                    raise ValueError("Use a still image, not an animation.")
                pic = ImageOps.exif_transpose(pic)
                if pic.mode in {"RGBA", "LA"} or "transparency" in pic.info:
                    rgba = pic.convert("RGBA")
                    pic = Image.alpha_composite(Image.new("RGBA", pic.size, "white"), rgba)
                return pic.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("Use an image of at most 12 megapixels.") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError("Could not decode this image. Try re-saving it as a PNG.") from None

def encode_png(rgb):
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()

def load_model(weights=WEIGHTS):
    model = ColorUNet()
    if weights.suffix == ".json":
        manifest = json.loads(weights.read_text(encoding="utf-8"))
        if manifest["format"] != "numpy-state-dict-fp16":
            raise ValueError("Unsupported model artifact format.")
        names = manifest["parts"]
        if not names or any(not isinstance(n,str) or Path(n).name != n or not n.endswith(".bin") for n in names):
            raise ValueError("Invalid model part names.")
        combined = b"".join((weights.parent / name).read_bytes() for name in names)
        if sha256(combined).hexdigest() != manifest["sha256"]:
            raise ValueError("Model checksum mismatch. Re-download the original files.")
        with np.load(BytesIO(combined),allow_pickle=False) as decoded:
            state = {name:torch.from_numpy(decoded[name].astype(np.float32)) for name in decoded.files}
        size = int(manifest["size"])
    else:
        artifact = torch.load(weights, map_location="cpu", weights_only=True)
        state, size = artifact["state"], int(artifact["size"])
    model.load_state_dict(state)
    if size < 16 or size > 512 or size % 2:
        raise ValueError("Unexpected model input size.")
    model.eval()
    return model, size

@torch.inference_mode()
def colorize(model, network_size, image, intensity=1.0):
    if not 0 <= intensity <= 1.5:
        raise ValueError("Color intensity must be between 0% and 150%.")
    picture = image.copy()
    picture.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    full = as_lab(picture)
    h, w = full.shape[:2]
    # Letterbox to the model's square input rather than stretch the photo.
    ratio = network_size / max(h, w)
    nh, nw = max(1, round(h*ratio)), max(1, round(w*ratio))
    y, x = (network_size-nh)//2, (network_size-nw)//2
    small = cv2.resize(full[:,:,0], (nw,nh), interpolation=cv2.INTER_AREA)
    tile = np.pad(small, ((y,network_size-nh-y),(x,network_size-nw-x)), mode="edge")
    tensor = torch.from_numpy(tile / 100)[None,None]
    chroma = model(tensor)[0].permute(1,2,0).numpy()[y:y+nh,x:x+nw] * (128 * intensity)
    chroma = cv2.resize(chroma, (w,h), interpolation=cv2.INTER_CUBIC)
    neutral = np.zeros_like(full)
    neutral[:,:,0] = full[:,:,0]
    result = neutral.copy()
    result[:,:,1:] = chroma.clip(-128, 127)
    return lab_as_rgb(neutral), lab_as_rgb(result)
