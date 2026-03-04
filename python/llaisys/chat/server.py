from __future__ import annotations

import json
import time
import uuid
from typing import List, Optional

from pydantic import BaseModel

from .service import ChatService, ChatRequest, ChatMessage

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
except Exception as exc:  # runtime import guard
    FastAPI = None
    JSONResponse = None
    StreamingResponse = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class MessageModel(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[MessageModel]
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 256
    stream: bool = False


def create_app(chat_service: ChatService):
    if FastAPI is None:
        raise RuntimeError(f"fastapi is required for chat server: {_IMPORT_ERROR}")

    app = FastAPI(title="LLAISYS Chat API")

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        chat_req = ChatRequest(
            request_id=request_id,
            messages=[ChatMessage(role=m.role, content=m.content) for m in req.messages],
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
            stream=req.stream,
        )
        chat_service.submit(chat_req)

        if req.stream:
            def event_stream():
                created = int(time.time())
                for ck in chat_service.stream(request_id):
                    payload = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "choices": [{"index": 0, "delta": {"content": ck.token_text}, "finish_reason": "stop" if ck.finished else None}],
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        resp = chat_service.wait(request_id)
        created = int(time.time())
        body = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": resp.text}, "finish_reason": "stop"}],
        }
        return JSONResponse(body)

    return app
