from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


svc_path = Path(__file__).resolve().parents[1] / "python" / "llaisys" / "chat" / "service.py"
spec = spec_from_file_location("llaisys_chat_service", svc_path)
service_mod = module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = service_mod
spec.loader.exec_module(service_mod)

ChatService = service_mod.ChatService
ChatRequest = service_mod.ChatRequest
ChatMessage = service_mod.ChatMessage


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class DummyMeta:
    end_token = 0


class DummyModel:
    def __init__(self):
        self.meta = DummyMeta()

    def reset_cache(self):
        pass

    def generate(self, inputs, max_new_tokens=1, temperature=0.8, top_k=40, top_p=0.95):
        last = inputs[-1] if inputs else 65
        nxt = 0 if last == 90 else min(last + 1, 90)
        return list(inputs) + [nxt]


def test_chat_service_wait_and_stream():
    svc = ChatService(model=DummyModel(), tokenizer=DummyTokenizer(), max_batch_size=2)
    req = ChatRequest(messages=[ChatMessage(role="user", content="A")], max_tokens=3)
    rid = svc.submit(req)

    resp = svc.wait(rid, timeout=3)
    assert len(resp.token_ids) >= 1

    rid2 = svc.submit(ChatRequest(messages=[ChatMessage(role="user", content="B")], max_tokens=2, stream=True))
    chunks = list(svc.stream(rid2))
    assert chunks
    assert chunks[-1].finished

    svc.shutdown()
