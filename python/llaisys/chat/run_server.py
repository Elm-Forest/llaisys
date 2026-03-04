from __future__ import annotations

import argparse

import uvicorn
from transformers import AutoTokenizer

from ..libllaisys import DeviceType
from ..models.qwen2 import Qwen2
from .service import ChatService
from .server import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    device = DeviceType.CPU if args.device == "cpu" else DeviceType.NVIDIA
    model = Qwen2(args.model, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    svc = ChatService(model=model, tokenizer=tokenizer)
    app = create_app(svc)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
