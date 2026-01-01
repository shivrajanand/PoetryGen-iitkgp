import os
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
# ------------------- CACHE ISSUE RESOLVRE FOR CLUSTER -------------------------
import unsloth
from unsloth.chat_templates import train_on_responses_only, standardize_sharegpt, get_chat_template
from unsloth import FastModel, FastLanguageModel
import gc
import torch
import pandas as pd
from datasets import Dataset
from transformers import EarlyStoppingCallback, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
import json
import sys

# os.makedirs(os.environ["UNSLOTH_COMPILED_CACHE"], exist_ok=True)
# os.environ["TEMP"] = os.environ["UNSLOTH_COMPILED_CACHE"]
# os.environ["TEMPDIR"] = os.environ["UNSLOTH_COMPILED_CACHE"]
os.environ["UNSLOTH_COMPILED_CACHE"] = "/nlsasfs/home/dibd/dibd-unified/iitkgp/jeetk/SHIVRAJ/PoetryGen-iitkgp/unsloth_cache"
os.makedirs(os.environ["UNSLOTH_COMPILED_CACHE"], exist_ok=True)


# ------------------- CONFIGURATION -------------------------

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
EPOCHS = 20
SAVE_STEPS = 50
LOG_STEPS = 10
EARLY_STOP_THRESHOLD = 1e-4
EARLY_STOP_PATIENCE = 5

RESUME_ADAPTER = True if sys.argv[3].strip()=='1' else False
ADAPTER_DIR = sys.argv[4] if sys.argv[3] else None

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOP_PATIENCE, early_stopping_threshold=EARLY_STOP_THRESHOLD)

# ---------------------- MODEL LOADING ---------------------------

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"{MODEL_NAME} Loaded Successfully!")
model = FastLanguageModel.get_peft_model(
    model,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj",    "gate_proj", "up_proj", "down_proj",],

    r=R,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print(f"PEFT {MODEL_NAME} Loaded Successfully!")

if RESUME_ADAPTER:
    print(f"Loading LoRA adapter from {ADAPTER_DIR}")
    model.load_adapter(
        ADAPTER_DIR,
        adapter_name="default",
        is_trainable=True
    )
    print("LoRA adapter loaded successfully")


# ---------------------- DATA PREP (FIXED) ---------------------------

df = pd.read_csv(TRAIN_DATA).dropna()
print(f"CSV LOADED. PATH = {TRAIN_DATA}")

def convert_to_conversations(df):
    data_list = []
    for _, row in df.iterrows():
        user_msg = (
            f"{row['prompt'].strip()}\n\n"
            "हिंदी अर्थ:\n"
            f"{row['meaning'].strip()}\n\n"
            "संस्कृत पद्य:\n"
        )

        data_list.append({
            "conversations": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": row["Shlok"]},
            ]
        })
    return data_list

raw_data = convert_to_conversations(df)
dataset = Dataset.from_list(raw_data)

tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
model.resize_token_embeddings(len(tokenizer))

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=["conversations"],
)

# def is_within_max_seq(example):
#     token_count = len(
#         tokenizer(
#             example["text"],
#             add_special_tokens=False
#         ).input_ids
#     )
#     return token_count <= MAX_SEQ

# original_len = len(dataset)
# dataset = dataset.filter(is_within_max_seq, num_proc=1)
# filtered_len = len(dataset)

# print(f"Filtered dataset: {original_len} -> {filtered_len}")
# print(f"Removed {original_len - filtered_len} examples exceeding {MAX_SEQ} tokens")

split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
eval_ds = split["test"]

print("Dataset created and split successfully.")
print("Train:", len(train_ds), "Eval:", len(eval_ds))

print("########################## Train-DS[0] ##########################")
print(train_ds[0]["text"])
print("#################################################################")


# ---------------------- TRAINER CONFIG ---------------------------

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    callbacks=[early_stopping_callback],
    max_seq_length=MAX_SEQ,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors='pt'),
    packing=False,
        dataset_map_kwargs={
        "num_proc": 1,
        "load_from_cache_file": False,
    },
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
        save_total_limit=3,
        save_strategy="steps",

        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        per_device_eval_batch_size=1,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        optim="adamw_torch",
        seed=42,
        dataloader_num_workers=0,
        report_to="tensorboard",

        warmup_steps=0,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        max_grad_norm=1.0,

        disable_tqdm=False,

        remove_unused_columns=True,
        dataset_text_field="text",
        packing=False,
        max_steps=10
    ),
)

# ---------------------- TRAINING STARTS ---------------------------
print(repr(train_ds[0]["text"][:200]))

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

print(f"TRAINING STARTS")

trainer_stats = trainer.train()

print(f"TRAINING ENDS")

trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print(f"BEST MODEL AND TOKENIZER SAVED IN {FINAL_DIR}")

# ---------------------- SAVING METRICS ---------------------------
summary_path = os.path.join(OUT_DIR, "TRAINING_SUMMARY_FULL.json")

state_dict = dict(trainer.state.__dict__)
# Optional: remove big log history
# state_dict.pop("log_history", None)

full_summary = {
    "model_name": MODEL_NAME,
    "training_args": trainer.args.to_dict(),
    "trainer_state": state_dict,
    "eval_metrics": trainer.evaluate(),
    "model_config": trainer.model.config.to_dict(),
}


with open(summary_path, "w") as f:
    json.dump(full_summary, f, indent=2)

print(f"Full training summary saved in {summary_path}.")


# ---------------------- CLEAN UP ---------------------------
del model, trainer
gc.collect()
torch.cuda.empty_cache()

print("ALL VARIABLES CLEANED")
print("SCRIPT ENDS HERE")
