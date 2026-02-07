from datasets import load_dataset
from transformers import AutoTokenizer
import os
from itertools import islice
import zstandard as zstd
import json

DATASET_DIR = os.environ["DATASET_DIR"]
OUT_DIR = os.path.join(DATASET_DIR, "c4_en", "raw")
SEQ_LEN = 512
os.makedirs(OUT_DIR, exist_ok=True)

target_tokens = 5_000_000_000
shard_docs = 200_000
batch_size = 32

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
it = iter(ds)

cctx = zstd.ZstdCompressor(level=3)

tokens = 0
docs = 0
shard_id = 0

def open_writer(shard_id):
    path = os.path.join(OUT_DIR, f"shard_{shard_id:03d}.json.zst")
    f = open(path, "wb")
    w = cctx.stream_writer(f)
    return path, f, w

path, f, w = open_writer(shard_id)

try:
    done = False
    while tokens < target_tokens and not done:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        
        texts = [ex["text"] for ex in batch]
        enc = tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_attention_mask=False
        )
        #batch_tokens = sum(len(x) for x in enc["input_ids"])
        for text, ids in zip(texts, enc["input_ids"]):
            n = len(ids)
            if n==0:
                continue

            if tokens + n > target_tokens:
                done = True
                break
            
            if docs > 0 and docs % shard_docs == 0:
                w.flush(zstd.FLUSH_FRAME)
                w.close()
                f.close()

                shard_id+=1
                path,f,w = open_writer(shard_id)
                
            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            w.write(line.encode("utf-8"))

            tokens += n 
            docs += 1

            if docs % 100_000 == 0:
                print(f"docs={docs:,} tokens={tokens:,} shard={shard_id} last_n={n}")

finally:
    try:
        w.flush(zstd.FLUSH_FRAME)
    except Exception:
        pass
    try:
        w.close()
    except Exception:
        pass
    try:
        f.close()
    except Exception:
        pass

meta = {
    "source":"allenai/c4",
    "config":"en",
    "format":"zst",
    "token_counter":"gpt2",
    "counting_rule":f"truncation=False",
    "target_tokens":target_tokens,
    "saved_token_est":tokens,
    "saved_docs":docs,
    "shard_docs":shard_docs,
    "batch_size":batch_size,
}

with open(os.path.join(OUT_DIR, "meta.json"), "w") as mf:
    json.dump(meta, mf, indent=2)

print("DONE")
print("OUT_DIR:", OUT_DIR)
print("tokens_est:", tokens)
print("docs:", docs)
print("shards", shard_id+1)
