from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import pandas as pd
import torch
import time

assert len(
    sys.argv) == 3, "USAGE: python inference-phi4.py <in-csv-path> <out-csv-path>"

csv_path = sys.argv[1]
out_path = sys.argv[2]

MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_PROMPT = """The goal is to generate a Sanskrit verse in anuṣṭubh meter from the given Hindi meaning.

RULES:
- The verse has 32 syllables (akshara) in total.
- The verse has 4 padas.
- Each pada has exactly 8 syllables.
- The verse is written in 2 lines, each line has 16 syllables.
- The 5th syllable of every pada must be LAGHU (short).
- The 6th syllable of every pada must be GURU (long).
- The 7th syllable of the 1st and 3rd pada must be DEERGHA.
- The 7th syllable of the 2nd and 4th pada must be HRASVA.

SYLLABLE RULES:
- LAGHU vowels: अ, इ, उ, ऋ, ऌ
- GURU vowels: आ, ई, ऊ, ॠ, ॡ, ए, ऐ, ओ, औ
- HRASVA vowels: अ, इ, उ, ऋ, ऌ
- DEERGHA vowels: आ, ई, ऊ, ॠ, ॡ, ए, ऐ, ओ, औ
- A syllable containing anusvāra (ं) or visarga (ः) is always GURU.
- A syllable followed by a conjunct consonant (saṁyuktākṣara) is always GURU.

Below are examples.

INPUT:
हिंदी अर्थ:
क्योंकि शत्रुविजयी राजालोग मर्यादाका उल्लंघन करनेवाले अपने पुत्र, नाती पोते और भाईका भी आदर नहीं करते

OUTPUT:
न हि पुत्रं न नप्तारं न भ्रातरमरिंदमाः
समतिक्रान्तमर्यादं पूजयन्ति नराधिपाः

INPUT:
हिंदी अर्थ:
उनके कटे हुए अङ्ग उसी प्रकार रक्तकी धारा बहते थे, जैसे वर्षा ऋतु में जल से भीगे हुए पर्वतों के शिखर जल की धारा बहाते हैं

OUTPUT:
तेषां छिन्नानि गात्राणि विसृजन्ति स्म शोणितम्
प्रावृषीवाभिवृष्टानि शृङ्गाण्यथ धराभृताम्

INPUT:
हिंदी अर्थ:
उस से आकाश व्याप्त है वह किसी से व्याप्त नहीं है वह बाहर और भीतर ठहरता है अखण्ड है और नित्य है

OUTPUT:
आकाशं तेन संव्याप्तं न तद्व्याप्तं च केनचित्
सबाह्याभ्यन्तरं तिष्ठत्यविच्छिन्नं निरन्तरम्
"""


tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4",
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda"
                                             ).to(DEVICE)
model.eval()

df = pd.read_csv(csv_path)

df["model-out"] = ""
total = len(df)
start_time = time.time()

def build_prompt(hindi_meaning: str) -> str:
    return (
        BASE_PROMPT
        + "\nINPUT:\n"
        + "हिंदी अर्थ:\n"
        + hindi_meaning.strip()
        + "\n\nOUTPUT:\n"
    )


with torch.no_grad():
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        prompt_text = build_prompt(row["meaning"])

        inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        # Optional: stop after two lines
        decoded = "\n".join(decoded.splitlines()[:2])

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
