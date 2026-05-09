import os
import csv
import json
import copy
import random
import hydra
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
import wandb
from sacrebleu.metrics import CHRF
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from utils import (
    grpo_generate_sequences,
    grpo_compute_loss_and_logs,
    compute_translation_metrics,
    LMScorer,
)
from dl import TranslationDataModule


def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort determinism (cuBLAS may still be non-deterministic).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _append_csv_row(path, row, fieldnames):
    is_new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _upload_dir_to_s3(local_dir, s3_uri_prefix):
    """Upload every file under local_dir to s3_uri_prefix preserving structure."""
    if not os.path.isdir(local_dir):
        return
    import boto3

    s3 = boto3.client("s3")
    assert s3_uri_prefix.startswith("s3://")
    bucket, _, key_prefix = s3_uri_prefix[len("s3://"):].partition("/")
    key_prefix = key_prefix.rstrip("/")
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
            key = f"{key_prefix}/{rel}" if key_prefix else rel
            s3.upload_file(local_path, bucket, key)
            print(f"  uploaded -> s3://{bucket}/{key}")


def _upload_file_to_s3(local_path, s3_uri):
    if not os.path.exists(local_path):
        return
    import boto3

    s3 = boto3.client("s3")
    assert s3_uri.startswith("s3://")
    bucket, _, key = s3_uri[len("s3://"):].partition("/")
    s3.upload_file(local_path, bucket, key)
    print(f"  uploaded -> s3://{bucket}/{key}")


def _sync_outputs_to_s3(s3_output_uri, exp_name, summary_json_path,
                        training_log_path, plots_dir, model_save_dir, upload_model):
    """Push per-experiment outputs under s3_output_uri/{exp_name}/."""
    if not s3_output_uri:
        return
    base = s3_output_uri.rstrip("/")
    target = f"{base}/{exp_name}"
    print(f"Syncing outputs to {target}/ ...")
    try:
        _upload_file_to_s3(summary_json_path, f"{target}/results/{exp_name}.json")
        _upload_file_to_s3(training_log_path, f"{target}/results/training_log_{exp_name}.csv")
        _upload_dir_to_s3(plots_dir, f"{target}/plots/{exp_name}")
        if upload_model and model_save_dir:
            _upload_dir_to_s3(model_save_dir, f"{target}/model")
        print("S3 sync done.")
    except Exception as exc:
        print(f"[warn] S3 sync failed: {exc}")


def _save_run_plots(history, plots_dir):
    """Save reward / chrf / bleu / ter / bertscore vs steps plots from history."""
    if not history:
        return
    os.makedirs(plots_dir, exist_ok=True)
    steps = [row["step"] for row in history]
    plot_specs = [
        ("reward", "reward_vs_steps.png", "Reward"),
        ("chrf++", "chrf_vs_steps.png", "chrF++"),
        ("bleu", "bleu_vs_steps.png", "BLEU"),
        ("ter", "ter_vs_steps.png", "TER"),
        ("bertscore", "bertscore_vs_steps.png", "BERTScore (F1 x100)"),
    ]
    for key, fname, ylabel in plot_specs:
        ys = [row.get(key) for row in history if row.get(key) is not None]
        xs = [row["step"] for row in history if row.get(key) is not None]
        if not ys:
            continue
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Optimizer step")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs steps")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, fname))
        plt.close()

def _run_evaluation(
    model: torch.nn.Module,
    tokenizer,
    dataloader,
    eval_cfg,
    *,
    tgt_lang_id: int,
    device: torch.device,
    max_new_tokens: int,
    split_name: str = "eval",
    step_idx: int = 0,
    total_training_steps: int = 0,
    wandb_run = None,
    wandb_table = None,
):
    if eval_cfg is None or dataloader is None:
        return {}

    requested_metrics = getattr(eval_cfg, "translation_metric", [])
    if isinstance(requested_metrics, str):
        requested_metrics = [requested_metrics]
    requested_metrics = [metric.lower() for metric in requested_metrics]

    was_training = model.training
    model.eval()

    metric_prefix = split_name.strip().replace(" ", "_") or "eval"
    loader = dataloader
    predictions = []
    references = []
    sample_records = []

    # Progress bar setup
    try:
        total_batches = len(loader)
    except TypeError:
        # Fallback if len(val_loader) is not available
        total_batches = None

    num_beams = int(getattr(eval_cfg, "num_beams", 1))
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        forced_bos_token_id=tgt_lang_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if num_beams > 1:
        generation_kwargs["early_stopping"] = True

    with torch.no_grad():
        with tqdm(total=total_batches, desc=f"Evaluating ({metric_prefix})", leave=False) as pbar:
            for idx, batch in enumerate(loader):
                encoder_inputs, ground_truths, source_texts, sample_ids = batch
                if isinstance(ground_truths, str):
                    ground_truths = [ground_truths]
                else:
                    ground_truths = list(ground_truths)
                sample_ids = [int(sid) for sid in (sample_ids if isinstance(sample_ids, (list, tuple)) else [sample_ids])]
                encoder_inputs = {k: v.to(device, non_blocking=True) for k, v in encoder_inputs.items()}
                generated = model.generate(
                    input_ids=encoder_inputs["input_ids"],
                    attention_mask=encoder_inputs.get("attention_mask"),
                    **generation_kwargs,
                )
                if generated.dim() == 1:
                    generated = generated.unsqueeze(0)
                
                for sample_idx, (reference, source_text) in enumerate(zip(ground_truths, source_texts)):
                    hypothesis = tokenizer.decode(generated[sample_idx], skip_special_tokens=True)
                    ref_text = reference if isinstance(reference, str) else str(reference)
                    predictions.append(hypothesis)
                    references.append(ref_text)
                    sample_records.append((ref_text, hypothesis, source_text, sample_ids[sample_idx], step_idx))
                if pbar.total is not None:
                    pbar.update(1)

    results = {}
    if predictions:
        compute_bertscore = "bertscore" in requested_metrics or not requested_metrics
        bertscore_lang = getattr(eval_cfg, "bertscore_lang", "en")
        # Always compute the full IEEE-paper metric set: BLEU, chrF++, TER, BERTScore.
        metric_dict = compute_translation_metrics(
            predictions,
            references,
            bertscore_lang=bertscore_lang,
            compute_bertscore=compute_bertscore,
        )
        for k, v in metric_dict.items():
            results[f"{metric_prefix}/{k}"] = float(v)

    if results:
        print(f"{split_name.capitalize()} evaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

    if wandb_run is not None and (results or sample_records):
        # log the sample records 5 times across training steps
        if step_idx % (total_training_steps / 10) == 0 and split_name == "eval":
            for reference, hypothesis, source_text, sample_id, step in sample_records:
                wandb_table.add_data(reference, hypothesis, sample_id, step)
        if split_name == "test":
            for reference, hypothesis, source_text, sample_id, step in sample_records:
                wandb_table.add_data(reference, hypothesis, sample_id, step)
        wandb_run.log({f"{split_name}/Translations": wandb_table})
        wandb_run.log(results)

    if was_training:
        model.train()

    # Strip prefix for ease of consumption by callers
    flat = {key.split("/", 1)[1]: value for key, value in results.items() if "/" in key}
    return {
        "metrics": flat,
        "predictions": predictions,
        "references": references,
        "sources": [r[2] for r in sample_records],
    }


@hydra.main(version_base=None, config_path="./configs/", config_name="train")
def train(config: DictConfig):
    _set_global_seed(int(config.seed))

    # Experiment mode: baseline_eval | rl_baseline | modified_rl
    exp_cfg = getattr(config.task, "experiment", None)
    exp_mode = str(getattr(exp_cfg, "mode", "rl_baseline")).strip() if exp_cfg is not None else "rl_baseline"
    exp_name = str(getattr(exp_cfg, "name", exp_mode)) if exp_cfg is not None else exp_mode
    reward_cfg = getattr(exp_cfg, "reward", None) if exp_cfg is not None else None
    reward_type = str(getattr(reward_cfg, "type", "baseline")) if reward_cfg is not None else "baseline"
    reward_lambdas = (
        float(getattr(reward_cfg, "lambda1", 0.7)) if reward_cfg is not None else 0.7,
        float(getattr(reward_cfg, "lambda2", 0.2)) if reward_cfg is not None else 0.2,
        float(getattr(reward_cfg, "lambda3", 0.1)) if reward_cfg is not None else 0.1,
    )
    lm_model_name = (
        str(getattr(reward_cfg, "lm_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
        if reward_cfg is not None
        else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    s3_output_uri = str(getattr(exp_cfg, "s3_output_uri", "")) if exp_cfg is not None else ""
    upload_model_to_s3 = (
        bool(getattr(exp_cfg, "upload_model_to_s3", False)) if exp_cfg is not None else False
    )

    if exp_mode == "baseline_eval":
        force_eval_only = True
    elif exp_mode == "rl_baseline":
        force_eval_only = False
        reward_type = "baseline"
    elif exp_mode == "modified_rl":
        force_eval_only = False
        reward_type = "modified"
    else:
        force_eval_only = False  # unknown -> defer to eval.only flag

    print("\n" + "=" * 65)
    print(f"  Experiment  : {exp_name}")
    print(f"  Mode        : {exp_mode}  ({'eval only' if force_eval_only else 'training + eval'})")
    print(f"  Model       : {config.task.model.name}")
    print(f"  Reward      : {reward_type}" + (
        f"  (λ={reward_lambdas[0]:.1f}·chrF++ + {reward_lambdas[1]:.1f}·BLEU + {reward_lambdas[2]:.1f}·LMScore)"
        if reward_type == "modified" else "  (chrF++ + BLEU)"
    ))
    print(f"  Source      : {config.task.data.source_lang}  →  Target: {config.task.data.target_lang}")
    train_file_disp = getattr(config.task.data, "train_file", None) or config.task.data.path
    print(f"  Train file  : {train_file_disp}")
    print("=" * 65 + "\n")

    # Device mapping: policy (NLLB) on cuda:1, reference+goldfish on cuda:0
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        aux_device = torch.device("cuda:0")
        policy_device = torch.device("cuda:1")
    else:
        # Fallback to single GPU/CPU
        aux_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_device = aux_device
    # Build model and tokenizer
    model, tokenizer = get_model(config)
    model.to(policy_device)
    source_lang_code = config.task.data.source_lang
    target_lang_code = config.task.data.target_lang
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang_code
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = target_lang_code

    def _tokenize_with_lang(texts, src_lang):
        if isinstance(texts, str):
            texts = [texts]
        previous_src_lang = getattr(tokenizer, "src_lang", None)
        if previous_src_lang is not None:
            tokenizer.src_lang = src_lang
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if previous_src_lang is not None:
            tokenizer.src_lang = previous_src_lang
        return encoded

    eval_cfg = getattr(config.task, "eval", None)
    eval_only = bool(getattr(eval_cfg, "only", False)) if eval_cfg is not None else False
    if force_eval_only:
        eval_only = True
    run_eval = bool(getattr(eval_cfg, "run", False)) if eval_cfg is not None else False
    if eval_only:
        run_eval = True
    run_training = not eval_only
    eval_every_n_opt_steps = 0
    if run_eval and eval_cfg is not None:
        eval_every_n_opt_steps = int(getattr(eval_cfg, "every_n_opt_steps", 0))
    
    test_cfg = getattr(config.task, "test", None)
    run_test = bool(getattr(test_cfg, "run", False)) if test_cfg is not None else False
    test_eval_cfg = None
    if run_test:
        if eval_cfg is not None:
            test_eval_cfg = OmegaConf.merge(eval_cfg, test_cfg)
        else:
            test_eval_cfg = test_cfg

    # Initialize Weights & Biases
    model_name_for_run = str(getattr(config.task.model, "name", "model")).replace("/", "-")
    run_name = f"{model_name_for_run}_outcome_batch_{config.task.training.batch_size}_src_tgt_src_{target_lang_code}_chrf"
    run_wandb = bool(getattr(config.task.training, "use_wandb", True))
    wandb_table = None
    wandb_run = None
    if run_wandb:
        wandb_run = wandb.init(project="grpo-translation-nllb-multi-domain", name=run_name)
        wandb_table = wandb.Table(columns=["reference", "generated", "sample_id", "step"], log_mode="INCREMENTAL")
    # Reference model (frozen copy of the policy)
    reference_name = getattr(config.task.model, "reference_name", None)
    ref_model = None
    if run_training:
        if reference_name is not None:
            ref_model = AutoModelForSeq2SeqLM.from_pretrained(
                reference_name,
                attn_implementation="flash_attention_2",
                dtype=model.dtype,
            ).to(aux_device)
        else:
            ref_model = copy.deepcopy(model).to(aux_device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # Data module
    illegal_token_mask = None
    data = TranslationDataModule(
        tokenizer=tokenizer,
        illegal_token_mask=illegal_token_mask,
        data_path=config.task.data.path,
        dataset_config_name=config.task.data.dataset_config_name,
        source_lang=config.task.data.source_lang,
        target_lang=config.task.data.target_lang,
        sort_by_length=False,
        train_batch_size=int(getattr(config.task.training, "batch_size", 1)),
        train_file=getattr(config.task.data, "train_file", None),
        valid_file=getattr(config.task.data, "valid_file", None),
        test_file=getattr(config.task.data, "test_file", None),
    )
    data.setup("fit")
    val_dataloader = data.val_dataloader() if run_eval else None
    test_dataloader = None
    if run_test:
        test_dataloader = data.test_dataloader()
        if test_dataloader is None:
            print("Test split not available; skipping test evaluation.")
            run_test = False
            test_eval_cfg = None
    total_training_steps = int(getattr(config.task.training, "epochs", 1)) * len(data.train_dataloader())
    # Training hyperparameters
    max_epochs = int(config.task.training.epochs)
    updates_per_batch = int(getattr(config.task.training, "updates_per_batch", 50))

    # Print dataset and training configuration so the run is self-documenting.
    _bs = int(getattr(config.task.training, "batch_size", 2))
    _upb = updates_per_batch
    _batches_per_epoch = len(data.train_data) // _bs + (1 if len(data.train_data) % _bs else 0)
    print(f"[data]  train={len(data.train_data)}  valid={len(data.val_data)}"
          + (f"  test={len(data.test_data)}" if data.test_data is not None else ""))
    if run_training:
        print(f"[train] epochs={max_epochs}  batch_size={_bs}  batches/epoch={_batches_per_epoch}")
        print(f"[train] updates_per_batch={_upb}  num_return_sequences={int(getattr(config.task.training, 'num_return_sequences', 4))}")
        print(f"[train] total optimizer steps ≈ {_batches_per_epoch * _upb * max_epochs}")
    grad_accum_steps = int(getattr(config.task.training, "accumulate_grad_batches", 1))
    if grad_accum_steps < 1:
        raise ValueError("accumulate_grad_batches must be >= 1.")
    num_return_sequences = int(getattr(config.task.training, "num_return_sequences", 4))
    max_new_tokens = int(getattr(config.task.constraints, "max_sentence_len", 128))
    gen_temperature = float(getattr(config.task.training, "gen_temperature", 1.3))
    beta = float(getattr(config.task.training, "beta", 0.04))
    clip_param = float(getattr(config.task.training, "clip_param", 0.2))
    tgt_lang_id = tokenizer.convert_tokens_to_ids(target_lang_code)
    src_lang_id = tokenizer.convert_tokens_to_ids(source_lang_code)

    # Lazily-loaded fluency scorer for the modified reward.
    lm_scorer = None
    if reward_type == "modified" and run_training:
        print(f"Loading LM scorer ({lm_model_name}) on {aux_device}...")
        lm_scorer = LMScorer(model_name=lm_model_name, device=str(aux_device))

    # Output directories per experiment
    repo_root = hydra.utils.get_original_cwd()
    results_dir = os.path.join(repo_root, "results")
    plots_dir = os.path.join(repo_root, "plots", exp_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    training_log_path = os.path.join(repo_root, "results", f"training_log_{exp_name}.csv")
    metric_history = []  # list of {"step", "reward", "bleu", "chrf++", "ter", "bertscore"}
    last_train_reward = float("nan")

    # Optimizer setup
    if run_training:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found for the optimizer.")
        optimizer = torch.optim.AdamW(
            trainable_params, lr=float(getattr(config.task.training, "lr", 7e-6))
        )
        optimizer.zero_grad()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        print(
            f"Trainable parameters: {trainable_params_count:,} "
            f"({trainable_params_count / max(total_params, 1):.2%}) out of {total_params:,}"
        )
        accum_steps_since_update = 0
        optimizer_step = 0
        
    def _evaluate_and_log(split_name, dataloader, cfg, step_idx_local, train_reward=float("nan")):
        if dataloader is None or cfg is None:
            return None
        out = _run_evaluation(
            model,
            tokenizer,
            dataloader,
            cfg,
            tgt_lang_id=tgt_lang_id,
            device=policy_device,
            max_new_tokens=max_new_tokens,
            split_name=split_name,
            step_idx=step_idx_local,
            total_training_steps=total_training_steps,
            wandb_run=wandb_run,
            wandb_table=wandb_table,
        )
        if out is None or not out.get("metrics"):
            return out
        if split_name == "eval":
            row = {
                "step": int(step_idx_local),
                "reward": float(train_reward),
                "bleu": float(out["metrics"].get("bleu", 0.0)),
                "chrf++": float(out["metrics"].get("chrf++", 0.0)),
                "ter": float(out["metrics"].get("ter", 0.0)),
                "bertscore": float(out["metrics"].get("bertscore", 0.0)),
            }
            metric_history.append(row)
            _append_csv_row(
                training_log_path,
                row,
                fieldnames=["step", "reward", "bleu", "chrf++", "ter", "bertscore"],
            )
        return out

    test_initial_out = None
    eval_initial_out = None
    if run_test:
        test_initial_out = _evaluate_and_log("test", test_dataloader, test_eval_cfg, 0)
    if run_eval:
        eval_initial_out = _evaluate_and_log("eval", val_dataloader, eval_cfg, 0)

    if run_training:
        for epoch in range(max_epochs):
            train_loader = data.train_dataloader()
            step_idx = 0

            for batch in train_loader:
                src_prompt, ground_truths, source_texts, sample_ids = batch
                encoder_inputs = {k: v.to(policy_device, non_blocking=True) for k, v in src_prompt.items()}
                batch_size = encoder_inputs["input_ids"].size(0)                
                while optimizer_step < updates_per_batch:
                    # Generate candidate sequences with current policy
                    generated_local = grpo_generate_sequences(
                        model,
                        tokenizer,
                        encoder_inputs,
                        tgt_lang_id,
                        max_new_tokens=max_new_tokens,
                        gen_temperature=gen_temperature,
                        num_return_sequences=num_return_sequences,
                        top_k=int(getattr(config.task.training, "top_k", 100)),
                        top_p=float(getattr(config.task.training, "top_p", 0.95)),
                        end_of_sentence_token_id=tokenizer.eos_token_id,
                        
                    )
                    
                    seq_len = generated_local.size(1)
                    try:
                        generated_all = generated_local.reshape(batch_size, num_return_sequences, seq_len)
                    except RuntimeError as exc:
                        raise RuntimeError(
                            "Unable to reshape generated sequences into (batch_size, num_return_sequences, seq_len). "
                            f"Batch size={batch_size}, num_return_sequences={num_return_sequences}, seq_len={seq_len}."
                        ) from exc
                    
                    # Prepare backward direction prompts from generated target sentences
                    forward_generated_flat = generated_all.reshape(
                        batch_size * num_return_sequences, seq_len
                    )
                    forward_prompt_texts = tokenizer.batch_decode(
                        forward_generated_flat, skip_special_tokens=True
                    )
                    
                    # Calculate chrF for forward translations (source → target)
                    chrf_metric = CHRF(word_order=2, char_order=6)
                    forward_references = [
                        ground_truths[idx // num_return_sequences]
                        for idx in range(len(forward_prompt_texts))
                    ]
                    forward_chrf_scores = [
                        chrf_metric.corpus_score(hypotheses=[hyp], references=[[ref]]).score
                        for hyp, ref in zip(forward_prompt_texts, forward_references)
                    ]
                    forward_chrf_mean = sum(forward_chrf_scores) / len(forward_chrf_scores)
                    
                    backward_inputs_encoded = _tokenize_with_lang(
                        forward_prompt_texts, target_lang_code
                    )
                    backward_inputs = {
                        k: v.to(policy_device, non_blocking=True)
                        for k, v in backward_inputs_encoded.items()
                    }
                    backward_batch_size = backward_inputs["input_ids"].size(0)
                    generated_backward = grpo_generate_sequences(
                        model,
                        tokenizer,
                        backward_inputs,
                        src_lang_id,
                        max_new_tokens=max_new_tokens,
                        gen_temperature=gen_temperature,
                        num_return_sequences=num_return_sequences,
                        top_k=int(getattr(config.task.training, "top_k", 100)),
                        top_p=float(getattr(config.task.training, "top_p", 0.95)),
                        end_of_sentence_token_id=tokenizer.eos_token_id,
                    )

                    back_seq_len = generated_backward.size(1)
                    try:
                        generated_backward_all = generated_backward.reshape(
                            backward_batch_size, num_return_sequences, back_seq_len
                        )
                    except RuntimeError as exc:
                        raise RuntimeError(
                            "Unable to reshape backward generated sequences into "
                            "(batch_size, num_return_sequences, seq_len). "
                            f"Batch size={backward_batch_size}, num_return_sequences={num_return_sequences}, "
                            f"seq_len={back_seq_len}."
                        ) from exc

                    expanded_source_texts = [
                        source_texts[idx // num_return_sequences]
                        for idx in range(backward_batch_size)
                    ]

                    loss_backward, logs_backward = grpo_compute_loss_and_logs(
                        model,
                        ref_model,
                        tokenizer,
                        backward_inputs,
                        generated_backward_all,
                        expanded_source_texts,
                        end_of_sentence_token_id=tokenizer.eos_token_id,
                        beta=beta,
                        clip_param=clip_param,
                        tgt_lang_id=src_lang_id,
                        reward_type=reward_type,
                        reward_lambdas=reward_lambdas,
                        lm_scorer=lm_scorer,
                    )

                    loss_scale = 1.0 / float(grad_accum_steps)
                    (loss_backward * loss_scale).backward()
                    accum_steps_since_update += 1
                    if accum_steps_since_update >= grad_accum_steps:
                        optimizer.step()
                        optimizer.zero_grad()
                        accum_steps_since_update = 0
                        optimizer_step += 1
                        if optimizer_step % int(getattr(config.task.training, "update_ref_policy_every_n_steps", 16)) == 0:
                            # Refresh reference model at epoch boundaries
                            ref_model = copy.deepcopy(model).to(aux_device)
                            ref_model.eval()
                            for p in ref_model.parameters():
                                p.requires_grad_(False)

                updates_per_batch += getattr(config.task.training, "updates_per_batch", 50)
                last_train_reward = float(logs_backward["reward"].item())
                if run_eval and eval_every_n_opt_steps > 0 and (optimizer_step % eval_every_n_opt_steps == 0):
                    _evaluate_and_log(
                        "eval",
                        val_dataloader,
                        eval_cfg,
                        optimizer_step,
                        train_reward=last_train_reward,
                    )

                print(
                    f"[epoch {epoch}] step {step_idx} (opt {optimizer_step}) | "
                    f"f_chrf={forward_chrf_mean:.4f} | "
                    f"b_loss={logs_backward['loss'].item():.4f} b_kl={logs_backward['kl'].item():.4f} "
                    f"b_reward={logs_backward['reward'].item():.4f} b_chrf={logs_backward['chrf'].item():.4f} "
                    f"b_bleu={logs_backward['bleu'].item():.4f} b_lm={logs_backward['lmscore'].item():.4f}"
                )
                # Print the reference and one generated sequence for inspection
                best_candidates = []
                for idx, reference in enumerate(ground_truths):
                    decoded = tokenizer.decode(
                        generated_all[idx, 0], skip_special_tokens=True
                    )
                    best_candidates.append((reference, decoded, source_texts[idx], sample_ids[idx]))
                ref_text, gen_text, src_text, ref_sample_id = best_candidates[0]
                print(f"Source[{ref_sample_id}]: {src_text}")
                print(f"Reference[{ref_sample_id}]: {ref_text}")
                print(f"Generated[{ref_sample_id}]: {gen_text}")
                back_best = tokenizer.decode(
                        generated_backward_all[0, 0], skip_special_tokens=True
                    )
                print(f"Back-Generated[{ref_sample_id}]: {back_best}")
                if run_wandb:
                    # Log to Weights & Biases
                    wandb.log(
                        {
                            "train/forward_chrf": float(forward_chrf_mean),
                            "train/backward_loss": float(logs_backward["loss"].item()),
                            "train/backward_kl": float(logs_backward["kl"].item()),
                            "train/backward_chrf": float(logs_backward["chrf"].item()),
                            "train/backward_reward": float(logs_backward["reward"].item()),
                            "train/backward_bleu": float(logs_backward["bleu"].item()),
                            "train/backward_lmscore": float(logs_backward["lmscore"].item()),
                        }
                    )
                    # if train_table is not None:
                    #     for reference, decoded, ref_sample_id in best_candidates:
                    #         train_table.add_data(
                    #             reference,
                    #             decoded,
                    #             ref_sample_id,
                    #         )
                    #     wandb.log({"train/Translations": train_table}, step=step_idx)
                step_idx += 1
            # Save the model every epoch
            epoch_save_dir = os.path.join(os.getcwd(), "model", f"{model_name_for_run}_epoch_{epoch}_{target_lang_code}_chrf")
            os.makedirs(epoch_save_dir, exist_ok=True)
            model.save_pretrained(epoch_save_dir)
            tokenizer.save_pretrained(epoch_save_dir)
            print(f"Model saved to {epoch_save_dir}")
    final_eval_out = None
    final_test_out = None
    final_step = int(locals().get("optimizer_step", 0))
    if run_eval and not eval_only:
        final_eval_out = _evaluate_and_log(
            "eval", val_dataloader, eval_cfg, final_step, train_reward=last_train_reward
        )

    if run_test:
        final_test_out = _evaluate_and_log("test", test_dataloader, test_eval_cfg, final_step)

    # Save plots and JSON summary
    _save_run_plots(metric_history, plots_dir)
    summary_path = os.path.join(results_dir, f"{exp_name}.json")
    summary = {
        "experiment": exp_name,
        "mode": exp_mode,
        "reward_type": reward_type,
        "reward_lambdas": list(reward_lambdas),
        "source_lang": source_lang_code,
        "target_lang": target_lang_code,
        "model": str(getattr(config.task.model, "name", "")),
        "seed": int(config.seed),
        "history": metric_history,
        "final_eval": (final_eval_out or eval_initial_out or {}).get("metrics") if (final_eval_out or eval_initial_out) else None,
        "final_test": (final_test_out or test_initial_out or {}).get("metrics") if (final_test_out or test_initial_out) else None,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary to {summary_path}")

    # CSV summary table (one row per experiment, for paper-friendly tables).
    csv_summary_path = os.path.join(results_dir, "summary.csv")
    csv_row = {
        "experiment": exp_name,
        "mode": exp_mode,
        "reward_type": reward_type,
    }
    for split_key, payload in (("eval", summary["final_eval"]), ("test", summary["final_test"])):
        if payload:
            for m in ("bleu", "chrf++", "ter", "bertscore"):
                csv_row[f"{split_key}_{m}"] = float(payload.get(m, 0.0))
    fieldnames = ["experiment", "mode", "reward_type"] + [
        f"{split}_{m}"
        for split in ("eval", "test")
        for m in ("bleu", "chrf++", "ter", "bertscore")
    ]
    _append_csv_row(csv_summary_path, csv_row, fieldnames=fieldnames)
    print(f"Appended CSV summary row to {csv_summary_path}")

    # Optional S3 sync of per-experiment outputs.
    last_model_dir = locals().get("epoch_save_dir")
    _sync_outputs_to_s3(
        s3_output_uri=s3_output_uri,
        exp_name=exp_name,
        summary_json_path=summary_path,
        training_log_path=training_log_path,
        plots_dir=plots_dir,
        model_save_dir=last_model_dir,
        upload_model=upload_model_to_s3,
    )

    if wandb.run is not None:
        wandb.finish()




def get_model(config: DictConfig):
    model_cfg = config.task.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg.name
    )

    use_lora = bool(getattr(model_cfg, "use_lora", False))
    if use_lora:
        for param in model.parameters():
            param.requires_grad_(False)

        lora_cfg = getattr(model_cfg, "lora", {})
        target_modules = tuple(getattr(lora_cfg, "target_modules", ("q_proj", "v_proj")))
        if not target_modules:
            raise ValueError("LoRA target modules must be a non-empty sequence of module suffixes.")

        replaced_modules = apply_lora(
            model=model,
            target_modules=target_modules,
            r=int(getattr(lora_cfg, "r", 8)),
            lora_alpha=float(getattr(lora_cfg, "alpha", 32)),
            lora_dropout=float(getattr(lora_cfg, "dropout", 0.05)),
        )

        print(f"LoRA enabled: adapted {len(replaced_modules)} modules with rank={int(getattr(lora_cfg, 'r', 8))}.")

    return model, tokenizer


if __name__ == "__main__":
    train()
