from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

SEQ_LEN = 512

cfg = GPT2Config()
model = GPT2LMHeadModel(cfg)

print(model)