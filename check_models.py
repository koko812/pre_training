from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
import time
import torch

SEQ_LEN = 512

with torch.device("meta"):
    st = time.perf_counter()
    cfg = GPT2Config()
    dt = time.perf_counter()
    model = GPT2LMHeadModel(cfg)
    ft = time.perf_counter()

#print(model)

n_params = sum(p.numel() for p in model.parameters())
print(n_params/1e6, "M params")

for m in model.children():
    print(m)
    
for m in model.modules():
    print(m)

for m in model.named_parameters():
    print(m)

for p in model.parameters():
    print(p.numel())

print(model)

print(f"conf: {dt-st}")
print(f"load: {ft-dt}")
print(f"total: {ft-st}")