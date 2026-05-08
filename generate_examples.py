"""Generate qualitative translation examples for paper tables.

Produces a CSV containing source, baseline, RL-baseline, and modified-RL
translations for inspection. Each model checkpoint argument can be either
a Hugging Face model id or a local directory.

Example:
    python generate_examples.py \
        --baseline_model facebook/nllb-200-distilled-600M \
        --rl_baseline_model runs/.../model/...rl_baseline... \
        --modified_rl_model runs/.../model/...modified_rl... \
        --test_file data/maithili_test.csv \
        --source_lang eng_Latn --target_lang mai_Deva \
        --num_examples 20 --output qualitative_examples.csv
"""
import argparse
import csv
import os

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset


def _load_test(path, source_lang, target_lang, n):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        ds = load_dataset("csv", data_files=path)["train"]
    elif ext == ".tsv":
        ds = load_dataset("csv", data_files=path, delimiter="\t")["train"]
    elif ext in (".json", ".jsonl"):
        ds = load_dataset("json", data_files=path)["train"]
    else:
        raise ValueError(f"Unsupported file extension '{ext}'")
    src_col = "sentence_" + source_lang
    tgt_col = "sentence_" + target_lang
    rows = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        rows.append((row[src_col], row.get(tgt_col, "")))
    return rows


@torch.no_grad()
def _translate(model, tokenizer, sources, source_lang, target_lang, device, max_new_tokens=128):
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang
    tgt_lang_id = tokenizer.convert_tokens_to_ids(target_lang)
    outputs = []
    for src in sources:
        enc = tokenizer(src, return_tensors="pt", truncation=True).to(device)
        gen = model.generate(
            **enc,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs.append(tokenizer.decode(gen[0], skip_special_tokens=True))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", type=str, required=True)
    parser.add_argument("--rl_baseline_model", type=str, default=None)
    parser.add_argument("--modified_rl_model", type=str, default=None)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--source_lang", type=str, default="eng_Latn")
    parser.add_argument("--target_lang", type=str, default="mai_Deva")
    parser.add_argument("--num_examples", type=int, default=20)
    parser.add_argument("--output", type=str, default="qualitative_examples.csv")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = _load_test(args.test_file, args.source_lang, args.target_lang, args.num_examples)
    sources = [e[0] for e in examples]
    refs = [e[1] for e in examples]

    rows = [{"source": s, "reference": r} for s, r in zip(sources, refs)]

    for col_name, ckpt in (
        ("baseline", args.baseline_model),
        ("rl_baseline", args.rl_baseline_model),
        ("modified_rl", args.modified_rl_model),
    ):
        if ckpt is None:
            for r in rows:
                r[col_name] = ""
            continue
        print(f"Loading {col_name} model from {ckpt}...")
        tok = AutoTokenizer.from_pretrained(ckpt)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device)
        mdl.eval()
        outs = _translate(
            mdl, tok, sources, args.source_lang, args.target_lang, device, args.max_new_tokens
        )
        for r, o in zip(rows, outs):
            r[col_name] = o
        del mdl, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fieldnames = ["source", "reference", "baseline", "rl_baseline", "modified_rl"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote {len(rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
