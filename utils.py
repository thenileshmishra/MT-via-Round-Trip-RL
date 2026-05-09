import math
from typing import Sequence, Optional, List

import torch
from sacrebleu.metrics import CHRF, BLEU, TER

# =========================
# Distributed GRPO Utilities
# =========================


class LMScorer:
    """Lightweight fluency reward via multilingual embedding similarity.

    Used as the LMScore component of the modified reward:
        R = lambda1 * chrF++ + lambda2 * BLEU + lambda3 * LMScore
    Score = (cosine_similarity + 1) / 2, mapped to [0, 1].
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def score(self, sources: Sequence[str], hypotheses: Sequence[str]):
        src_emb = self.model.encode(
            list(sources), convert_to_tensor=True, show_progress_bar=False
        )
        hyp_emb = self.model.encode(
            list(hypotheses), convert_to_tensor=True, show_progress_bar=False
        )
        cos = torch.nn.functional.cosine_similarity(src_emb, hyp_emb)
        return ((cos + 1.0) / 2.0).clamp(0.0, 1.0).cpu().tolist()


class GoldfishScorer:
    """Log-probability scoring via Goldfish KenLM models (Magnusson et al., 2023).

    Provides Table-2 fluency metric: average natural-log-probability per word.
    Falls back gracefully if kenlm or the model file is unavailable.

    NLLB lang code → ISO 639-3 used by Goldfish:
      mai_Deva → mai,  bho_Deva → bho,  asm_Beng → asm
    """

    _NLLB_TO_GOLDFISH = {
        "mai_Deva": "mai",
        "bho_Deva": "bho",
        "asm_Beng": "asm",
    }

    def __init__(self, nllb_lang_code: str):
        self.available = False
        self._model = None
        try:
            import kenlm
            from huggingface_hub import hf_hub_download

            lang = self._NLLB_TO_GOLDFISH.get(
                nllb_lang_code, nllb_lang_code.split("_")[0].lower()
            )
            model_path = hf_hub_download(
                repo_id="cis-lmu/Goldfish",
                filename=f"{lang}/{lang}.binary",
            )
            self._model = kenlm.Model(model_path)
            self.available = True
            print(f"[GoldfishScorer] Loaded KenLM model for '{lang}'")
        except Exception as exc:
            print(f"[GoldfishScorer] Unavailable for '{nllb_lang_code}': {exc}")

    def score(self, texts: List[str]) -> List[float]:
        """Return average natural-log-probability per word for each text."""
        if not self.available:
            return [float("nan")] * len(texts)
        scores = []
        for text in texts:
            words = max(len(text.split()), 1)
            log10_total = self._model.score(text, bos=True, eos=True)
            scores.append(log10_total * math.log(10) / words)
        return scores


def compute_translation_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    bertscore_lang: str = "en",
    compute_bertscore: bool = True,
):
    """Compute BLEU, chrF++, TER, BERTScore on a list of hyps/refs.

    Returns a dict with keys: bleu, chrf++, ter, bertscore, bertscore_p, bertscore_r
    (all on a 0-100 scale).
    """
    predictions = list(predictions)
    references = list(references)
    bleu = BLEU().corpus_score(predictions, [references]).score
    chrf = CHRF(word_order=2, char_order=6).corpus_score(predictions, [references]).score
    ter_score = TER().corpus_score(predictions, [references]).score
    metrics = {
        "bleu": float(bleu),
        "chrf++": float(chrf),
        "ter": float(ter_score),
    }
    if compute_bertscore:
        try:
            from bert_score import score as bert_score_fn

            P, R, F1 = bert_score_fn(
                predictions, references, lang=bertscore_lang, verbose=False
            )
            metrics["bertscore_p"] = float(P.mean().item()) * 100.0
            metrics["bertscore_r"] = float(R.mean().item()) * 100.0
            metrics["bertscore"] = float(F1.mean().item()) * 100.0
        except Exception as exc:  # pragma: no cover - optional dep
            print(f"[warn] BERTScore failed ({exc}); reporting 0.0")
            metrics["bertscore_p"] = 0.0
            metrics["bertscore_r"] = 0.0
            metrics["bertscore"] = 0.0
    else:
        metrics["bertscore_p"] = 0.0
        metrics["bertscore_r"] = 0.0
        metrics["bertscore"] = 0.0
    return metrics

@torch.no_grad()
def grpo_generate_sequences(
    model,
    tokenizer,
    encoder_inputs,
    tgt_lang_id,
    max_new_tokens: int,
    gen_temperature: float,
    num_return_sequences: int,
    top_k: int = 100,
    top_p: float = 0.9,
    end_of_sentence_token_id: int = None,
):
    eos_id = (
        end_of_sentence_token_id
        if end_of_sentence_token_id is not None
        else tokenizer.eos_token_id
    )
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=gen_temperature,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_id,
        top_k=top_k,
        top_p=top_p,
    )
    gen = model.generate(
        input_ids=encoder_inputs["input_ids"],
        attention_mask=encoder_inputs.get("attention_mask", None),
        forced_bos_token_id=tgt_lang_id,
        **generation_kwargs,
    )
    return gen


def _gather_log_probs_from_logits_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = logits.log_softmax(dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return gathered


def grpo_compute_decoder_per_token_logps(
    model,
    tokenizer,
    encoder_inputs,
    decoder_input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    device = next(model.parameters()).device

    # Repeat encoder inputs to match number of sequences
    base_batch_size = encoder_inputs["input_ids"].size(0)
    batch_multiplier = decoder_input_ids.size(0) // base_batch_size
    if batch_multiplier * base_batch_size != decoder_input_ids.size(0):
        raise ValueError(
            "decoder_input_ids size does not align with encoder batch size. "
            f"Got encoder batch {base_batch_size} and decoder batch {decoder_input_ids.size(0)}."
        )
    repeated_input_ids = encoder_inputs["input_ids"].repeat_interleave(batch_multiplier, dim=0)
    repeated_input_ids = repeated_input_ids.to(device)
    attention_mask = encoder_inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.repeat_interleave(batch_multiplier, dim=0)
        attention_mask = attention_mask.to(device)

    decoder_input_ids = decoder_input_ids.to(device)
    target_ids = target_ids.to(device)

    decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id).long()
    decoder_attention_mask = decoder_attention_mask.to(device)

    outputs = model(
        input_ids=repeated_input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (B, L, V)
    del outputs
    per_token_logps = _gather_log_probs_from_logits_logits(logits, target_ids)  # (B, L)
    return per_token_logps

def grpo_compute_loss_and_logs(
    model,
    ref_model,
    tokenizer,
    encoder_inputs,
    generated_sequences: torch.Tensor,
    ground_truths: Sequence[str],
    *,
    end_of_sentence_token_id: int,
    beta: float,
    clip_param: float,
    tgt_lang_id: int,
    reward_type: str = "baseline",
    reward_lambdas: Sequence[float] = (0.7, 0.2, 0.1),
    lm_scorer: Optional["LMScorer"] = None,
):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    else:
        ground_truths = list(ground_truths)

    if generated_sequences.dim() != 3:
        raise ValueError(
            "generated_sequences must be a 3D tensor of shape (batch_size, num_candidates, seq_len). "
            f"Received tensor with shape {tuple(generated_sequences.shape)}."
        )

    device = next(model.parameters()).device
    generated_sequences = generated_sequences.to(device)

    batch_size, num_candidates, seq_len = generated_sequences.size()
    if batch_size != len(ground_truths):
        raise ValueError(
            f"Number of ground truths ({len(ground_truths)}) does not match generated batch size ({batch_size})."
        )
    flat_sequences = generated_sequences.reshape(batch_size * num_candidates, seq_len)

    # Prepare decoder inputs/targets
    decoder_input_ids = flat_sequences[:, :-1]
    target_ids = flat_sequences[:, 1:]

    # Compute per-token logps under current and reference policies
    per_token_logps = grpo_compute_decoder_per_token_logps(
        model, tokenizer, encoder_inputs, decoder_input_ids, target_ids
    )
    with torch.no_grad():
        ref_per_token_logps = grpo_compute_decoder_per_token_logps(
            ref_model, tokenizer, encoder_inputs, decoder_input_ids, target_ids
        ).to(device)

    # Completion mask to ignore pads and tokens after first EOS
    is_pad = target_ids == tokenizer.pad_token_id
    is_eos = target_ids == end_of_sentence_token_id
    is_lang_id = target_ids == tgt_lang_id
    eos_cumsum = is_eos.cumsum(dim=-1)
    after_eos = eos_cumsum >= 1
    completion_mask = (~is_pad) & (~after_eos) & (~is_lang_id)
    completion_mask = completion_mask.reshape(batch_size, num_candidates, -1)

    # Decode generated sequences without special tokens for reward computation
    generated_texts = tokenizer.batch_decode(
        torch.where(
            generated_sequences.reshape(-1, generated_sequences.size(-1)) == tokenizer.pad_token_id,
            end_of_sentence_token_id,
            generated_sequences.reshape(-1, generated_sequences.size(-1)),
        ).cpu(),
        skip_special_tokens=True,
    )
    references = [
        ground_truths[idx // num_candidates]
        for idx in range(len(generated_texts))
    ]

    # Compute chrF with evaluate (character order=6) for outcome rewards
    chrf_metric = CHRF(word_order=2, char_order=6)
    bleu_metric = BLEU(effective_order=True)
    chrf_scores = []
    bleu_scores = []
    for hyp, ref in zip(generated_texts, references):
        chrf_scores.append(chrf_metric.corpus_score(hypotheses=[hyp], references=[[ref]]).score / 100.0)
        bleu_scores.append(bleu_metric.corpus_score(hypotheses=[hyp], references=[[ref]]).score / 100.0)
    chrf_scores_tensor = torch.tensor(chrf_scores, device=device)
    bleu_scores_tensor = torch.tensor(bleu_scores, device=device)
    chrf_mean = chrf_scores_tensor.mean()
    bleu_mean = bleu_scores_tensor.mean()

    if reward_type == "modified":
        # R = lambda1*chrF++ + lambda2*BLEU + lambda3*LMScore
        if lm_scorer is None:
            raise ValueError("reward_type='modified' requires an LMScorer instance.")
        lm_scores = lm_scorer.score(references, generated_texts)
        lm_scores_tensor = torch.tensor(lm_scores, device=device)
        l1, l2, l3 = reward_lambdas
        combined_scores_tensor = (
            l1 * chrf_scores_tensor + l2 * bleu_scores_tensor + l3 * lm_scores_tensor
        )
        lm_mean = lm_scores_tensor.mean()
    else:
        # Original paper baseline reward: R = chrF++ + BLEU
        combined_scores_tensor = chrf_scores_tensor + bleu_scores_tensor
        lm_mean = torch.tensor(0.0, device=device)

    rewards = combined_scores_tensor.reshape(batch_size, num_candidates)
    standardized_rewards = (rewards - rewards.mean(dim=1, keepdim=True)) / (
        rewards.std(dim=1, keepdim=True) + 1e-4
    )

    # Outcome-based advantages replicate the standardized reward over all tokens
    per_token_logps = per_token_logps.reshape(batch_size, num_candidates, -1)
    advantages = standardized_rewards.unsqueeze(-1).expand_as(per_token_logps)

    # PPO-style ratio (on-policy baseline trick)
    ratio = torch.exp(per_token_logps - per_token_logps.detach())
    clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
    per_token_adv_loss = torch.min(ratio * advantages, clipped_ratio * advantages)

    # KL penalty versus reference policy
    ref_minus_pi = ref_per_token_logps.reshape(batch_size, num_candidates, -1) - per_token_logps
    per_token_kl = torch.exp(ref_minus_pi) - ref_minus_pi - 1.0

    per_token_obj = per_token_adv_loss - beta * per_token_kl
    per_token_obj = per_token_obj * completion_mask

    # Mean over valid tokens per sequence, then batch mean
    token_counts = completion_mask.sum(dim=-1).clamp_min(1)
    # per_token
    seq_losses = -per_token_obj.sum(dim=-1) / token_counts
    loss = seq_losses.mean()

    with torch.no_grad():
        kl_mean = (per_token_kl * completion_mask).sum() / token_counts.sum()

    logs = {
        "loss": loss.detach(),
        "kl": kl_mean.detach(),
        "reward": rewards.mean().detach(),
        "chrf": chrf_mean.detach(),
        "bleu": bleu_mean.detach(),
        "lmscore": lm_mean.detach(),
    }
    return loss, logs
