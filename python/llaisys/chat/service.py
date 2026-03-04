from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatRequest:
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    stream: bool = False
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex}")


@dataclass
class ChatChunk:
    request_id: str
    token_id: int
    token_text: str
    finished: bool = False


@dataclass
class ChatResponse:
    request_id: str
    text: str
    token_ids: List[int]


class _RequestState:
    def __init__(self, req: ChatRequest, input_ids: List[int]):
        self.req = req
        self.input_ids = list(input_ids)
        self.generated: List[int] = []
        self.finished = False
        self.chunks: "queue.Queue[ChatChunk]" = queue.Queue()
        self.done = threading.Event()


class ChatService:
    """Single-model inference service with continuous scheduling loop.

    This implementation uses iteration-level scheduling and streams one token per active
    request in each scheduling round.
    """

    def __init__(self, model, tokenizer, max_batch_size: int = 8, poll_interval_s: float = 0.002):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.poll_interval_s = poll_interval_s
        self._waiting: "queue.Queue[_RequestState]" = queue.Queue()
        self._active: List[_RequestState] = []
        self._states: Dict[str, _RequestState] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def shutdown(self):
        self._stop.set()
        self._worker.join(timeout=1.0)

    def submit(self, req: ChatRequest) -> str:
        prompt = self._render_prompt(req.messages)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        st = _RequestState(req=req, input_ids=input_ids)
        with self._lock:
            self._states[req.request_id] = st
        self._waiting.put(st)
        return req.request_id

    def wait(self, request_id: str, timeout: Optional[float] = None) -> ChatResponse:
        st = self._states[request_id]
        ok = st.done.wait(timeout=timeout)
        if not ok:
            raise TimeoutError(f"request {request_id} timeout")
        text = self.tokenizer.decode(st.generated, skip_special_tokens=True)
        return ChatResponse(request_id=request_id, text=text, token_ids=st.generated)

    def stream(self, request_id: str) -> Iterable[ChatChunk]:
        st = self._states[request_id]
        while True:
            if st.done.is_set() and st.chunks.empty():
                break
            try:
                ck = st.chunks.get(timeout=0.1)
                yield ck
                if ck.finished:
                    break
            except queue.Empty:
                if st.done.is_set():
                    break

    def _run(self):
        # Note: current backend keeps a single KV-cache per model instance.
        # We therefore process each request in isolated cache windows by reset + replay,
        # while still applying iteration-level scheduler semantics.
        while not self._stop.is_set():
            self._fill_active()
            if not self._active:
                time.sleep(self.poll_interval_s)
                continue

            next_round: List[_RequestState] = []
            for st in self._active:
                if st.finished:
                    continue

                self.model.reset_cache()
                context = st.input_ids + st.generated
                if not context:
                    st.finished = True
                    st.done.set()
                    continue

                # One iteration decode
                next_token = self.model.generate(
                    context,
                    max_new_tokens=1,
                    temperature=st.req.temperature,
                    top_k=st.req.top_k,
                    top_p=st.req.top_p,
                )[-1]
                st.generated.append(int(next_token))

                token_text = self.tokenizer.decode([int(next_token)], skip_special_tokens=False)
                is_end = int(next_token) == int(getattr(self.model.meta, "end_token", -1))
                limit = len(st.generated) >= st.req.max_tokens
                finished = is_end or limit

                st.chunks.put(ChatChunk(request_id=st.req.request_id, token_id=int(next_token), token_text=token_text, finished=finished))
                if finished:
                    st.finished = True
                    st.done.set()
                else:
                    next_round.append(st)

            self._active = next_round

    def _fill_active(self):
        while len(self._active) < self.max_batch_size:
            try:
                st = self._waiting.get_nowait()
                self._active.append(st)
            except queue.Empty:
                break

    @staticmethod
    def _render_prompt(messages: List[ChatMessage]) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m.role}: {m.content}")
        parts.append("assistant:")
        return "\n".join(parts)
