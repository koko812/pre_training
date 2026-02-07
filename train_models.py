from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, TrainingArguments, Trainer
import time
import torch
from load_dataset import ZstJsonlPackedTokenDataset
import wandb
import os 
from datetime import datetime

SEQ_LEN = 512
MODEL_DIR = os.environ["MODEL_DIR"]
DATASET_BASE_DIR = os.environ["DATASET_DIR"]
RAW_DIR = os.path.join(os.environ["DATASET_DIR"], "c4_en", "raw")

class PackedCollator:
    def __call__(self, features):
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        assert all(len(f["input_ids"]) == SEQ_LEN for f in features)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


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

train_ds = ZstJsonlPackedTokenDataset(RAW_DIR)


timestamp = datetime.now().strftime("%Y%m%d_%H%M")
run_name_default = f"_run_{timestamp}"
run_name = "gpt2_fuck" + run_name_default
wandb.init(project="ultra_pretrain", name=run_name)

micro_bsz = 16
grad_accum = 2
max_steps = 30000
lr = 2e-4

args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, run_name),
    run_name=run_name,
    report_to="wandb",
    
    do_train=True,
    
    max_steps=max_steps,
    per_device_train_batch_size=micro_bsz,
    gradient_accumulation_steps=grad_accum,
    
    learning_rate=lr,
    weight_decay=0.1,
    warmup_steps=max_steps//100,
    lr_scheduler_type="cosine",
    
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    
    bf16=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=False,
    
    remove_unused_columns=False
)

trainer = Trainer(
    model = model,
    args=args,
    train_dataset=train_ds,
    data_collator=PackedCollator()
)

trainer.train()