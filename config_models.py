from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
import time
import torch

SEQ_LEN = 512

cfg = GPT2Config(n_layer=16, n_head=16)

with torch.device("meta"):
    model = GPT2LMHeadModel(cfg)

print(model)
n_params = sum(p.numel() for p in model.parameters())
print(n_params/1e6, "M params")


st = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
ft = time.perf_counter()
print(f"total: {ft-st}")