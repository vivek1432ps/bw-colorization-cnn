"""Luma Lab: local-only grayscale colorization workbench."""
from base64 import b64encode
from email.parser import BytesParser
from email.policy import default
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from time import perf_counter

import torch

from core import ROOT, WEIGHTS, MAX_UPLOAD, colorize, encode_png, load_model, read_image

SAMPLE = ROOT / "assets" / "astronaut-demo.jpg"
MAX_REQUEST = MAX_UPLOAD+65536

STYLE = """
*{box-sizing:border-box}
:root{color-scheme:dark}
body{margin:0;background:#100f17;color:#f5f2fd;font:15px/1.5 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:25px 32px 38px}
header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #383148;padding:0 0 18px}
.logo{font-weight:850;letter-spacing:.14em;color:#ffcda2}
.tag{background:#272235;border:1px solid #55465c;border-radius:100px;padding:4px 12px;color:#eddbff;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.085em}
.title{padding:22px 0 20px}
.kicker{color:#e8a5bc;letter-spacing:.12em;font-size:11px;text-transform:uppercase;font-weight:800}
h1{margin:4px 0 9px;font-size:clamp(35px,6vw,64px);line-height:1.03;letter-spacing:-.065em}
p{margin:0}.muted{color:#c6bed4}.layout{display:grid;grid-template-columns:minmax(224px,.72fr) minmax(0,2fr);gap:16px}
.card{padding:17px;background:#1b1725;border:1px solid #41354c;border-radius:15px;min-width:0}
h2{font-size:18px;margin:0 0 13px}.hint{color:#bfb1ce;font-size:12px;margin:6px 0 0}
label{display:block;font-weight:760;margin-top:14px;font-size:13px}
input{width:100%;margin:5px 0 0;padding:7px;color:#eee3ff;background:#130f1b;border:1px solid #554765;border-radius:8px;font:13px system-ui}
input:focus,button:focus,summary:focus,a:focus{outline:2px solid #ffc3df;outline-offset:3px}
button,.save{width:100%;display:inline-block;margin-top:13px;padding:10px;border-radius:8px;text-align:center;font:800 13px system-ui;cursor:pointer;text-decoration:none}
button{border:0;color:#241423;background:#ffbdcd}
button.secondary,.save{background:#29213a;color:#efdfff;border:1px solid #594968}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:10px}
figure{margin:0;min-width:0}figcaption{margin-bottom:6px;color:#d8c6ef;font-size:12px;font-weight:800;letter-spacing:.065em;text-transform:uppercase}
figure img{width:100%;aspect-ratio:1;object-fit:contain;display:block;background:#0f0c15;border:1px solid #4c3e56;border-radius:10px}
.meta{margin-top:10px;color:#c7b3dc;font-size:12px}
.foot{margin-top:15px}.error{margin-bottom:9px;color:#ffc2cc;font-weight:750}
@media(max-width:760px){main{padding:16px}.layout{grid-template-columns:1fr}.card{padding:13px}}
"""

def uri(png):
    return "data:image/png;base64,"+b64encode(png).decode("ascii")

def html_result(pair):
    before, after, duration = pair
    return f'''
    <div class="panels">
      <figure><figcaption>01 / Before</figcaption><img alt="Grayscale input" src="{uri(before)}"></figure>
      <figure><figcaption>02 / After</figcaption><img alt="CNN colorized output" src="{uri(after)}"></figure>
    </div><p class="meta">CPU inference · {duration:.2f} s in this run · Suggested colors, not historical fact</p>
    <a class="save" href="{uri(after)}" download="luma-lab-colorized.png">Download colorized PNG</a>'''

def page(pair=None, message=None, intensity=100):
    pieces = html_result(pair) if pair else '<p class="muted">Upload a photo or try the built-in example to preview the result.</p>'
    error = f'<p class="error" role="alert">{escape(message)}</p>' if message else ""
    return f'''<!doctype html><html lang="en"><head><meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>Luma Lab | CNN Colorizer</title>
    <style>{STYLE}</style></head><body><main>
    <header><div class="logo">◐ &nbsp; LUMA LAB</div><span class="tag">Local CPU · LAB U-Net</span></header>
    <section class="title"><div class="kicker">Computer vision workbench / 03</div>
    <h1>Color from grayscale.</h1><p class="muted">A tiny U-Net proposes colors while retaining the input’s lightness structure.</p></section>
    {error}<div class="layout">
    <section class="card"><h2>Image settings</h2>
      <form action="/" method="post" enctype="multipart/form-data">
        <label for="photo">Upload a photo</label><input type="file" name="photo" id="photo" accept="image/jpeg,image/png,image/webp" required>
        <p class="hint">PNG, JPEG, or WebP. Up to 8 MB and 12 MP.</p>
        <label for="intensity">Color strength (%)</label>
        <input type="number" id="intensity" name="intensity" min="0" max="150" value="{intensity}" required>
        <button type="submit">Colorize image →</button>
      </form>
      <form action="/" method="get"><button class="secondary" type="submit">Load the astronaut example</button></form>
      <p class="hint">Local inference. Photos are processed in memory; the app does not save or upload them.</p>
    </section>
    <section class="card"><h2>Side-by-side preview</h2>{pieces}</section></div>
    <p class="hint foot">The model was trained on small CIFAR-10 images; color bleeding and muted or inaccurate hues are expected, especially on out-of-domain photos.</p>
    </main></body></html>'''

def infer(model, size, payload, strength):
    image = read_image(payload)
    start = perf_counter()
    gray, color = colorize(model,size,image,strength/100)
    return encode_png(gray),encode_png(color),perf_counter()-start

def unpack(body, content_type):
    if not content_type.startswith("multipart/form-data;") or "\r" in content_type or "\n" in content_type:
        raise ValueError("Choose a photo using the upload form.")
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n"
    envelope = BytesParser(policy=default).parsebytes(header.encode("ascii")+body)
    if not envelope.is_multipart():
        raise ValueError("Upload data is malformed.")
    fields = {}
    for part in envelope.iter_parts():
        Name = part.get_param("name",header="content-disposition")
        if Name in {"photo","intensity"} and Name not in fields and not part.is_multipart():
            fields[Name] = part.get_payload(decode=True)
    try:
        strength = int(fields.get("intensity",b"100"))
    except (TypeError, ValueError):
        raise ValueError("Color strength must be an integer from 0 to 150.") from None
    if not 0<=strength<=150:
        raise ValueError("Color strength must be an integer from 0 to 150.")
    return fields.get("photo") or b"",strength

class Handler(BaseHTTPRequestHandler):
    def send_page(self, code, markup):
        body=markup.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Content-Security-Policy","default-src 'none'; img-src data:; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path!="/":
            return self.send_error(404)
        self.send_page(200,page(self.server.sample))

    def do_POST(self):
        if self.path!="/":
            return self.send_error(404)
        try:
            size=int(self.headers.get("Content-Length","0"))
        except ValueError:
            size=0
        if not 0<size<=MAX_REQUEST:
            self.close_connection=True
            return self.send_page(413,page(self.server.sample,"Choose a photo no larger than 8 MB."))
        try:
            self.connection.settimeout(8)
            payload,strength=unpack(self.rfile.read(size), self.headers.get("Content-Type",""))
            result=infer(*self.server.model,payload,strength)
        except (ValueError,UnicodeError,TimeoutError) as exc:
            self.close_connection=True
            return self.send_page(400,page(self.server.sample,str(exc)))
        self.send_page(200,page(result,intensity=strength))

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8000)
    parser.add_argument("--weights",type=type(WEIGHTS),default=WEIGHTS)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if not args.weights.exists():
        parser.error("Model checkpoint missing. See README for training.")
    with ThreadingHTTPServer(("127.0.0.1",args.port),Handler) as server:
        server.model=load_model(args.weights)
        server.sample=infer(*server.model,SAMPLE.read_bytes(),100)
        print(f"Open http://127.0.0.1:{args.port}",flush=True)
        server.serve_forever()

if __name__=="__main__":
    main()
