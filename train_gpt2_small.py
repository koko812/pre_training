from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
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
    n_embd=768,
    n_positions=SEQ_LEN,
    n_layer=12,
    n_head=12,
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
max_steps = 305000
lr = 2e-4

lr_scheduler_kwargs={"min_lr": lr*0.1, "num_cycles": 0.5}

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
    lr_scheduler_type="cosine_warmup_with_min_lr",
    lr_scheduler_kwargs=lr_scheduler_kwargs,
    warmup_steps=max_steps//100,
    
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    
    bf16=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=False,
    
    remove_unused_columns=False
)


class SimpleGenCallback(TrainerCallback):
    def __init__(self, tokenizer, max_new_tokens=120):
        self.tok = tokenizer
        self.max_new_tokens = max_new_tokens
        self.prompts = [
            "The meaning of life is",
            "In distant future, humans and AI",
            "Write a short paragraph about Japan",
            "Transformer is",
            "Once upon a time"
        ]

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step != 1 and state.global_step % 1000 != 0:
            return

        model = kwargs["model"]
        model.eval()
        rows = []
        
        with torch.no_grad():
            for p in self.prompts :
                inputs = self.tok(p, return_tensors="pt").to(model.device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
                text = self.tok.decode(out[0], skip_special_tokens=True)
                rows.append([state.global_step, p, text])
            
        table = wandb.Table(columns=["step", "prompt", "output"])
        for r in rows:
            table.add_data(*r)
        
        wandb.log({"samples": table}, step=state.global_step)

        model.train()


class TokenSpeedCallback(TrainerCallback):
    def __init__(self, seq_len, micro_bsz, grad_accum):
        self.seq_len = seq_len
        self.micro_bsz = micro_bsz
        self.grad_accum = grad_accum
        
        self.total_tokens = 0
        self.t_last = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t_last = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        t = time.perf_counter()
        dt = t - self.t_last if self.t_last is not None else None
        self.t_last = t

        tokens_this_step = self.seq_len * self.micro_bsz * self.grad_accum
        self.total_tokens += tokens_this_step

        if dt and state.global_step % args.logging_steps == 0:
            wandb.log(
                {
                    "train/total_tokens": self.total_tokens,
                    "train/tokens_per_sec": tokens_this_step / dt,
                    "train/sec_per_step": dt
                },
                step=state.global_step
            )

trainer = Trainer(
    model = model,
    args=args,
    train_dataset=train_ds,
    data_collator=PackedCollator(),
    callbacks=[SimpleGenCallback(tokenizer), TokenSpeedCallback(seq_len=SEQ_LEN, micro_bsz=micro_bsz, grad_accum=grad_accum)]
)

trainer.train()