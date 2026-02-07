from datasets import load_dataset
from transformers import AutoTokenizer
import os
import zstandard as zstd

DATASET_DIR = os.environ["DATASET_DIR"]
OUT_DIR = os.path.join(DATASET_DIR, "c4_en", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

target_docs = 5_000_000_000
shard_docs = 1_000_000
batch_size = 256

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
it = iter(ds)

target_tokens = 5_000_000
seen = 0

for ex in ds:
    ids = tokenizer.encode(ex["text"])
    seen += len(ids)
    print(seen)
    if seen > target_tokens:
        break

print("tokens")