import os, json
import zstandard as zstd
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

class ZstJsonlTextDataset(IterableDataset):
    def __init__(self, raw_dir, chunk_size=1 << 20):
        self.raw_dir = raw_dir
        self.chunk_size = chunk_size
        
    def _iter_jsonl_bytes(self, path):
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as r:
                buf = b""
                while True:
                    chunk = r.read(self.chunk_size)
                    if not chunk:
                        break
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        if line:
                            yield line
                if buf:
                    yield buf

    def __iter__(self):
        files = sorted(
            os.path.join(self.raw_dir, n)
            for n in os.listdir(self.raw_dir)
            if n.endswith("json.zst")
        )
        for fp in files:
            for raw in self._iter_jsonl_bytes(fp):
                ex = json.loads(raw)
                text = ex.get("text", "")
                if text:
                    yield {"text": text}
                    

class ZstJsonlTokenDataset(IterableDataset):
    def __init__(self, raw_dir,  tokenizer_name="gpt2", max_chars=20000, chunk_size=1 << 20):
        self.raw_dir = raw_dir
        self.chunk_size = chunk_size
        self.max_chars = max_chars
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tok.pad_token = self.tok.eos_token
        
    def _iter_jsonl_bytes(self, path):
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as r:
                buf = b""
                while True:
                    chunk = r.read(self.chunk_size)
                    if not chunk:
                        break
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        if line:
                            yield line
                if buf:
                    yield buf

    def __iter__(self):
        files = sorted(
            os.path.join(self.raw_dir, n)
            for n in os.listdir(self.raw_dir)
            if n.endswith("json.zst")
        )
        for fp in files:
            for raw_line in self._iter_jsonl_bytes(fp):
                ex = json.loads(raw_line.decode("utf-8"))
                text = ex.get("text", "")
                if not text:
                    return
                if self.max_chars is not None and len(text) > self.max_chars:
                    continue

                ids = self.tok(text, add_special_tokens=False)["input_ids"]
                ids.append(self.tok.eos_token_id)
                if not ids:
                    continue

                yield {"input_ids": ids}
                    
class ZstJsonlPackedTokenDataset(IterableDataset):
    def __init__(self, raw_dir,  tokenizer_name="gpt2", block_size=512, max_chars=20000, chunk_size=1 << 20):
        self.raw_dir = raw_dir
        self.chunk_size = chunk_size
        self.max_chars = max_chars
        self.block_size = block_size
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tok.pad_token = self.tok.eos_token
        
    def _iter_jsonl_bytes(self, path):
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as r:
                buf = bytearray()
                while True:
                    chunk = r.read(self.chunk_size)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        if line:
                            yield line
                if buf:
                    yield buf

    def __iter__(self):
        files = sorted(
            os.path.join(self.raw_dir, n)
            for n in os.listdir(self.raw_dir)
            if n.endswith("json.zst")
        )

        eos = self.tok.eos_token_id
        buffer = []

        for fp in files:
            for raw_line in self._iter_jsonl_bytes(fp):
                ex = json.loads(raw_line.decode("utf-8"))
                text = ex.get("text", "")
                if not text:
                    continue

                if self.max_chars is not None and len(text) > self.max_chars:
                    continue

                #ids = self.tok(text, add_special_tokens=False)["input_ids"]
                ids = self.tok(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.block_size - 1,  # eos を足す余地
                )["input_ids"]


                if not ids:
                    continue

                if ids[-1] != eos:
                    ids.append(self.tok.eos_token_id)

                buffer.extend(ids)


                while len(buffer) >= self.block_size:
                    out = buffer[:self.block_size]
                    buffer = buffer[self.block_size:]
                    yield {"input_ids": out, "labels": out}


if __name__ == "__main__":
    print("===== raw =====")
    RAW_DIR = os.path.join(os.environ["DATASET_DIR"], "c4_en", "raw")
    ds = ZstJsonlTextDataset(RAW_DIR)

    for i,d in enumerate(ds):
        print(i, len(d["text"]), d["text"][:80].replace("\n"," "))
        if i > 3:
            break

    print("===== tokens =====")
    RAW_DIR = os.path.join(os.environ["DATASET_DIR"], "c4_en", "raw")
    ds = ZstJsonlTokenDataset(RAW_DIR)

    for i,d in enumerate(ds):
        print(i, len(d["input_ids"]), d["input_ids"][:5], d["input_ids"][-5:])
        if i > 3:
            break

    print("===== packed =====")
    RAW_DIR = os.path.join(os.environ["DATASET_DIR"], "c4_en", "raw")
    ds = ZstJsonlPackedTokenDataset(RAW_DIR)

    for i,d in enumerate(ds):
        eos_id = 50256
        eos_cnt = 0
        for id in d["input_ids"]:
            if id == eos_id:
                eos_cnt+=1
        print(i, len(d["input_ids"]), d["input_ids"][:5], d["input_ids"][-5:], f"eos_cnt: {eos_cnt}")
        if i > 3:
            break