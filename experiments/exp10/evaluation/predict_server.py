"""
Serve the trained model over HTTP, so that tools running in another Python environment (the LeRobot one,
which drives the arm) can ask it for predictions.

    python evaluation/predict_server.py --model outputs/qwen35_08b_v2/final --port 8765

POST /predict with JSON {"target": "blue ball", "left_png": <base64>, "right_png": <base64>}
-> {"joints": [pan, lift, elbow, wrist_flex] (degrees, SIMULATOR convention), "text": raw answer, "ms": time}
GET /health -> {"ok": true, "model": ...}
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
from evaluation.evaluate import load_trained  # noqa: E402
from training.train import chat_prompt  # noqa: E402

STATE = {}


def predict(target, left, right):
    model, processor = STATE["model"], STATE["processor"]
    prompt = common.PROMPT.format(descr=target)
    enc = processor(text=[chat_prompt(processor, prompt)], images=[[left, right]], return_tensors="pt",
                    padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                             pad_token_id=processor.tokenizer.pad_token_id)
    text = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    pred = common.parse_answer(text)
    return (None if pred is None else [float(v) for v in pred]), text


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, {"ok": True, "model": STATE["path"]})
        else:
            self._reply(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path != "/predict":
            self._reply(404, {"error": "unknown path"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            imgs = [Image.open(io.BytesIO(base64.b64decode(req[k]))).convert("RGB") for k in ("left_png", "right_png")]
            t0 = time.time()
            joints, text = predict(req["target"], *imgs)
            self._reply(200, {"joints": joints, "text": text, "ms": round(1000 * (time.time() - t0))})
        except Exception as e:                                      # report, keep serving
            self._reply(500, {"error": repr(e)})

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    STATE["model"], STATE["processor"] = load_trained(args.model)
    STATE["path"] = args.model
    print(f"model loaded; serving on http://{args.host}:{args.port}  (POST /predict, GET /health)")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
