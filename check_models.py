from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
import time

SEQ_LEN = 512

st = time.perf_counter()
cfg = GPT2Config()
dt = time.perf_counter()
model = GPT2LMHeadModel(cfg)
ft = time.perf_counter()

print(model)

print(f"conf: {dt-st}")
print(f"load: {ft-dt}")
print(f"total: {ft-st}")