from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
import time
import torch

SEQ_LEN = 512

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
cfg = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_embd=512,
    n_positions=SEQ_LEN,
    n_layer=6,
    n_head=8,
)
model = GPT2LMHeadModel(cfg)
model.resize_token_embeddings(len(tokenizer))

print(len(tokenizer))