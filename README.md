# Arabic-to-Japanese Neural Machine Translation

Fine-tuning of Meta's **NLLB-200** (600M parameters) for **Arabic → Japanese** translation using **LoRA**, trained entirely on **CPU-only hardware**.

## Overview

Direct Arabic–Japanese parallel data is scarce, so this project builds a training set from JW300 by pivoting through English, then fine-tunes NLLB-200 with parameter-efficient LoRA adapters. Full fine-tuning of the 600M-parameter model crashed the kernel on CPU, so LoRA was used to make training feasible without a GPU.

## Highlights

- Arabic → Japanese translation with a fine-tuned NLLB-200
- LoRA adapters targeting the attention `q_proj` and `v_proj` layers
- Trains on CPU-only hardware
- Parallel corpus built from JW300 using English as a pivot language

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- pandas

## Repository Contents

```
├── Arabic-to-Japanese-Neural-Machine-Translation/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors     # LoRA adapter weights
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_args.bin
└── README.md
```

## Data Pipeline

1. Load the JW300 Arabic–English and Japanese–English parallel corpora.
2. Merge them on the shared English sentence to form Arabic–Japanese pairs.
3. Deduplicate on the English pivot column to avoid a many-to-many merge explosion.
4. Filter out NaN and non-string rows before and after saving to CSV.

## Training Setup

| Setting             | Value                |
|---------------------|----------------------|
| Base model          | NLLB-200 (600M)      |
| Method              | LoRA                 |
| LoRA target modules | `q_proj`, `v_proj`   |
| Hardware            | CPU only             |
| Source language     | Arabic (`arb_Arab`)  |
| Target language     | Japanese (`jpn_Jpan`)|

To fit CPU memory, training uses a reduced batch size and mid-training evaluation is disabled.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

base_model_name = "facebook/nllb-200-distilled-600M"
adapter_path = "Arabic-to-Japanese-Neural-Machine-Translation"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, src_lang="arb_Arab")
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_path)

text = "Your Arabic sentence here"
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("jpn_Jpan"),
    max_length=128,
)
print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])
```

## Challenges Solved

- **MemoryError** from a many-to-many pandas merge: fixed by deduplicating on the English pivot column.
- **Kernel crash** during full 600M-parameter fine-tuning: fixed with LoRA, a smaller batch size, and disabling mid-training evaluation.
- **Tokenizer TypeError** caused by NaN rows in the JW300 data: fixed by string filtering before and after CSV export.

## Results

| Metric | Value |
|--------|-------|
| BLEU   | XX    |

## Author

**Rahma Mahmoud Hassan Mohamed**
AI & Data Science Engineer

- GitHub: [RahmaEl-Sheikh](https://github.com/RahmaEl-Sheikh)
- LinkedIn: [linkedin.com/in/rahmaelsheikh](https://linkedin.com/in/rahmaelsheikh)
