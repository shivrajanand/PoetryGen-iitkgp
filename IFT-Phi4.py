# ================================================================
# MUST BE FIRST
# ================================================================
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd
import torch
import json
import gc
import sys
import os
import unsloth  # must be first

os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["UNSLOTH_COMPILED_CACHE"] = (
    "/nlsasfs/home/dibd/dibd-unified/iitkgp/jeetk/SHIVRAJ/"
    "PoetryGen-iitkgp/unsloth_cache"
)
os.makedirs(os.environ["UNSLOTH_COMPILED_CACHE"], exist_ok=True)

# ================================================================
# Imports
# ================================================================


# ================================================================
# CONFIG
# ================================================================
MODEL_NAME = "microsoft/phi-4"

TRAIN_DATA = sys.argv[1]
OUT_DIR = sys.argv[2]
FINAL_DIR = f"{OUT_DIR}/FINAL"

MAX_SEQ = 1024
R = 16
ALPHA = 32
DROPOUT = 0.05
LR = 1e-6

BATCH = 1
GRAD_ACC = 16
EPOCHS = 30

SAVE_STEPS = 50
LOG_STEPS = 10

EARLY_STOP_THRESHOLD = 1e-4
EARLY_STOP_PATIENCE = 5

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOP_PATIENCE, early_stopping_threshold=EARLY_STOP_THRESHOLD)

# ================================================================
# MODEL
# ================================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ,
    load_in_16bit=True,
    full_finetuning=False,
)

tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

model = FastLanguageModel.get_peft_model(
    model,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    r=R,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    use_gradient_checkpointing="unsloth",
)

print("Model loaded")

# ================================================================
# DATA PREP (PURE PYTHON ONLY)
# ================================================================
df = pd.read_csv(TRAIN_DATA).dropna()

examples = []

for _, row in df.iterrows():
    user_msg = (
        f"{row['prompt'].strip()}\n\n"
        "हिंदी अर्थ:\n"
        f"{row['meaning'].strip()}\n\n"
        "संस्कृत पद्य:\n"
    )

    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": row["Shlok"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ,
        padding=False,
    )

    examples.append({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].copy(),
    })

# Train / eval split (pure python)
split_idx = int(0.9 * len(examples))
train_data = examples[:split_idx]
eval_data = examples[split_idx:]

train_ds = Dataset.from_list(train_data)
eval_ds = Dataset.from_list(eval_data)

print(f"Dataset ready | Train={len(train_ds)} Eval={len(eval_ds)}")

# ================================================================
# TRAINER
# ================================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    callbacks=[early_stopping_callback],
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    ),
    dataset_text_field=None,  # CRITICAL
    packing=False,
    args=SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=False,
        bf16=True,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        save_total_limit=3,
        save_strategy="steps",
        eval_strategy="steps",
        per_device_eval_batch_size=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        seed=42,
        warmup_steps=0,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        max_grad_norm=1.0,
        disable_tqdm=False,
        report_to="tensorboard",
        # max_steps=10,
    ),
)

# trainer = train_on_responses_only(
#     trainer,
#     instruction_part="<|im_start|>user<|im_sep|>",
#     response_part="<|im_start|>assistant<|im_sep|>",
# )

# ================================================================
# TRAIN
# ================================================================
print("TRAINING STARTS")
trainer.train()
print("TRAINING ENDS")

trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

# ================================================================
# CLEANUP
# ================================================================
del trainer, model
gc.collect()
torch.cuda.empty_cache()

print("DONE")
