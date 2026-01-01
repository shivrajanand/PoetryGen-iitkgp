from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import pandas as pd
import torch
import time
from pathlib import Path
assert len(
    sys.argv) == 4, "USAGE: python inference-phi4.py <in-csv-path> <out-csv-path> <adapter-path>"

csv_path = sys.argv[1]
out_path = sys.argv[2]
ADAPTER_PATH = sys.argv[3]
ADAPTER_PATH = ADAPTER_PATH if Path(ADAPTER_PATH).is_dir() else None
MAX_NEW_TOKENS = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4",
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda"
                                             )

if ADAPTER_PATH:
    print("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
model.eval()

df = pd.read_csv(csv_path)

df["model-out"] = ""
total = len(df)
start_time = time.time()

with torch.no_grad():
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        text = (
            f"{row['prompt'].strip()}\n\n"
            "हिंदी अर्थ:\n"
            f"{row['meaning'].strip()}\n\n"
            "संस्कृत पद्य:\n"
        )
        
        # print(f"=*50\nINPUT:\n{text}")
        
        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to(DEVICE)
        
        # print(f"=*50\nTokenized Inputs:\n{inputs}")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        df.at[idx, "model-out"] = decoded
        
        if i % 10 == 0 or i == total:
            elapsed = time.time() - start_time
            avg = elapsed / i
            eta = avg * (total - i)

            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Processed {i}/{total} rows | "
                f"Elapsed: {elapsed:.1f}s | "
                f"ETA: {eta:.1f}s",
                flush=True
            )
            
            df.to_csv(out_path, index=False)

df.to_csv(out_path, index=False)
