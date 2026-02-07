import torch
from transformers import GPT2Config, GPT2LMHeadModel

def count_params(module):
    return sum(p.numel() for p in module.parameters())

def breakdown_gpt2(model):
    # 1) token embedding
    wte = model.transformer.wte
    wte_params = count_params(wte)

    # 2) position embedding
    wpe = model.transformer.wpe
    wpe_params = count_params(wpe)

    # 3) transformer blocks
    blocks = model.transformer.h
    n_layer = len(blocks)
    per_block = count_params(blocks[0]) if n_layer > 0 else 0
    blocks_total = count_params(blocks)

    # 4) final LayerNorm
    ln_f = model.transformer.ln_f
    ln_f_params = count_params(ln_f)

    # 5) lm_head
    lm_head = model.lm_head
    lm_head_params = count_params(lm_head)

    total = count_params(model)

    print("===== GPT-2 Parameter Breakdown =====")
    print(f"1) token embedding (wte): {wte_params/1e6:.2f} M")
    print(f"2) position embedding (wpe): {wpe_params/1e6:.2f} M")
    print(f"3) transformer blocks: {n_layer} × {per_block/1e6:.2f} M = {blocks_total/1e6:.2f} M")
    print(f"4) final LayerNorm (ln_f): {ln_f_params/1e6:.4f} M")
    print(f"5) lm_head: {lm_head_params/1e6:.2f} M")
    print("--------------------------------------")
    print(f"Total: {total/1e6:.2f} M")

# 例
cfg = GPT2Config(n_layer=16, n_head=16, n_embd=768)
with torch.device("meta"):
    model = GPT2LMHeadModel(cfg)

breakdown_gpt2(model)
