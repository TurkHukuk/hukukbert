#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 TurkHukuk.ai


import argparse
import json
import math
import sys
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

Z_95 = 1.959963984540054

def load_dataset(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Basic validation
                raw_sentence = item.get("sentence", "")
                if not isinstance(raw_sentence, str):
                    logger.warning(f"Line {i+1}: Invalid sentence. Skipping.")
                    continue
                sentence = raw_sentence.replace("[mask]", "[MASK]")
                if sentence.count("[MASK]") != 1:
                    logger.warning(f"Line {i+1}: Expected exactly one [MASK]. Skipping.")
                    continue
                options = item.get("options")
                gold = item.get("gold")
                if not isinstance(options, list) or not options or not all(isinstance(x, str) and x.strip() for x in options):
                    logger.warning(f"Line {i+1}: Invalid options. Skipping.")
                    continue
                if not isinstance(gold, str) or not gold.strip():
                    logger.warning(f"Line {i+1}: Invalid gold. Skipping.")
                    continue
                item["sentence"] = sentence
                items.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Line {i+1}: Invalid JSON. Skipping.")
    for idx, item in enumerate(items):
        doc_id = item.get("id") or item.get("doc_id") or f"ITEM_{idx+1:04d}"
        item["_doc_id"] = doc_id
    return items

def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return device_arg


def wilson_interval(successes: int, total: int, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score confidence interval for a Bernoulli proportion."""
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    z2 = z * z
    denom = 1.0 + (z2 / total)
    center = (phat + (z2 / (2.0 * total))) / denom
    spread = (z / denom) * math.sqrt((phat * (1.0 - phat) / total) + (z2 / (4.0 * total * total)))
    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    return (low, high)


def _binom_pmf_half(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(log_comb - (n * math.log(2.0)))


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """
    Two-sided exact McNemar p-value using the Binomial(b+c, 0.5) test.
    b: A correct / B wrong
    c: A wrong / B correct
    """
    n = b + c
    if n <= 0:
        return 1.0
    tail = 0.0
    limit = min(b, c)
    for i in range(limit + 1):
        tail += _binom_pmf_half(n, i)
    return min(1.0, 2.0 * tail)


def mcnemar_chi2_cc_pvalue(b: int, c: int) -> Tuple[float, float]:
    """Continuity-corrected chi-square approximation for McNemar (df=1)."""
    n = b + c
    if n <= 0:
        return (0.0, 1.0)
    chi2 = ((abs(b - c) - 1.0) ** 2) / n
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return (chi2, p_value)


def compare_top1_with_mcnemar(
    model_a: str,
    model_b: str,
    top1_hits_a: Dict[str, bool],
    top1_hits_b: Dict[str, bool],
) -> Optional[Dict[str, Any]]:
    common_ids = sorted(set(top1_hits_a.keys()) & set(top1_hits_b.keys()))
    if not common_ids:
        return None

    both_correct = 0
    a_only = 0
    b_only = 0
    both_wrong = 0
    for doc_id in common_ids:
        a_hit = bool(top1_hits_a[doc_id])
        b_hit = bool(top1_hits_b[doc_id])
        if a_hit and b_hit:
            both_correct += 1
        elif a_hit and not b_hit:
            a_only += 1
        elif (not a_hit) and b_hit:
            b_only += 1
        else:
            both_wrong += 1

    exact_p = mcnemar_exact_pvalue(a_only, b_only)
    chi2_cc, chi2_p = mcnemar_chi2_cc_pvalue(a_only, b_only)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_common": len(common_ids),
        "a_only": a_only,
        "b_only": b_only,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "exact_p": exact_p,
        "chi2_cc": chi2_cc,
        "chi2_cc_p": chi2_p,
    }

def prepare_masked_input(sentence: str, num_masks: int, tokenizer: PreTrainedTokenizer) -> str:
    """
    Replaces [MASK] with exactly `num_masks` mask tokens.
    """
    mask_token = tokenizer.mask_token
    # Create a string of N mask tokens
    replacement = " ".join([mask_token] * num_masks)
    return sentence.replace("[MASK]", replacement, 1)


def should_prepend_space(tokenizer: PreTrainedTokenizer) -> bool:
    """Return True only for tokenizers where leading-space markers matter."""
    model_type = (getattr(tokenizer, "model_type", "") or "").lower()
    if model_type in {"roberta", "gpt2", "bart"}:
        return True

    backend = getattr(tokenizer, "backend_tokenizer", None)
    pre_tokenizer = getattr(backend, "pre_tokenizer", None)
    if pre_tokenizer is None:
        return False

    cls_name = pre_tokenizer.__class__.__name__.lower()
    return "bytelevel" in cls_name


def ensure_padding_token(tokenizer: PreTrainedTokenizer) -> bool:
    """
    Ensure tokenizer has a padding token.
    Returns True if vocab was expanded (new token added), else False.
    """
    if tokenizer.pad_token_id is not None:
        return False

    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return False
    if tokenizer.sep_token is not None:
        tokenizer.pad_token = tokenizer.sep_token
        return False
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return False

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return True


def build_option_map(
    tokenizer: PreTrainedTokenizer,
    sentence: str,
    options: List[str],
) -> List[Tuple[str, List[int]]]:
    """Tokenize options with tokenizer-aware whitespace handling."""
    needs_space = " [MASK]" in sentence
    use_prefix_space = needs_space and should_prepend_space(tokenizer)

    option_map: List[Tuple[str, List[int]]] = []
    for opt in options:
        clean_opt = opt.strip()
        if use_prefix_space and not clean_opt.startswith(" "):
            opt_to_tokenize = " " + clean_opt
        else:
            opt_to_tokenize = clean_opt

        opt_ids = tokenizer.encode(opt_to_tokenize, add_special_tokens=False)
        if not opt_ids:
            continue
        option_map.append((opt, opt_ids))
    return option_map


def cap_pll_variant_batch_size(
    requested_batch_size: int,
    max_seq_length: int,
    vocab_size: int,
    device: str,
) -> int:
    """Return a backend-safe upper bound for PLL variant batch size."""
    int_max = 2_147_483_647
    safe_cap = int_max // max(1, max_seq_length * max(1, vocab_size))
    if device == "mps":
        safe_cap = max(1, safe_cap // 2)
    return max(1, min(requested_batch_size, max(1, safe_cap)))


def encode_masked_sentence(
    tokenizer: PreTrainedTokenizer,
    sentence: str,
    num_masks: int,
    max_seq_length: int,
) -> Dict[str, List[int]] | None:
    """
    Encode sentence with `num_masks` mask tokens while trying to preserve mask positions
    under truncation.
    """
    masked_sentence = prepare_masked_input(sentence, num_masks, tokenizer)
    fast = tokenizer(
        masked_sentence,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
    )
    input_ids = fast.get("input_ids", [])
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is not None and sum(1 for tok in input_ids if tok == mask_token_id) == num_masks:
        return fast

    # Fallback: build a mask-centered window to avoid losing [MASK] during truncation.
    if "[MASK]" not in sentence or mask_token_id is None:
        return None

    left, right = sentence.split("[MASK]", 1)
    left_ids = tokenizer.encode(left, add_special_tokens=False)
    right_ids = tokenizer.encode(right, add_special_tokens=False)
    special = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_seq_length - special - num_masks
    if budget < 0:
        return None

    left_take = min(len(left_ids), budget // 2)
    right_take = min(len(right_ids), budget - left_take)
    remaining = budget - left_take - right_take
    if remaining > 0:
        add_left = min(remaining, len(left_ids) - left_take)
        left_take += add_left
        remaining -= add_left
    if remaining > 0:
        add_right = min(remaining, len(right_ids) - right_take)
        right_take += add_right

    core_ids = left_ids[-left_take:] + [mask_token_id] * num_masks + right_ids[:right_take]
    final_input_ids = tokenizer.build_inputs_with_special_tokens(core_ids)
    result: Dict[str, List[int]] = {
        "input_ids": final_input_ids,
        "attention_mask": [1] * len(final_input_ids),
    }
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(core_ids)
    if token_type_ids is not None and len(token_type_ids) == len(final_input_ids):
        result["token_type_ids"] = token_type_ids
    return result


def batched_vocab_predictions(
    model,
    tokenizer: PreTrainedTokenizer,
    batch_items: List[Dict[str, object]],
    device: str,
    max_seq_length: int,
    topk: int = 3,
) -> List[List[Tuple[str, float]]]:
    """Compute top-k vocab predictions for each item in one batched forward pass."""
    model.eval()
    per_item: List[List[Tuple[str, float]]] = [[] for _ in batch_items]
    features: List[Tuple[int, Dict[str, List[int]]]] = []
    for idx, item in enumerate(batch_items):
        sentence = str(item["sentence"])
        enc = encode_masked_sentence(tokenizer, sentence, 1, max_seq_length)
        if enc is not None:
            features.append((idx, enc))

    if not features:
        return per_item

    with torch.inference_mode():
        model_inputs = [enc for _, enc in features]
        inputs = tokenizer.pad(model_inputs, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs).logits
        input_ids = inputs["input_ids"]
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            return per_item
        mask_matrix = input_ids == mask_token_id

        for row, (item_idx, _) in enumerate(features):
            mask_positions = torch.where(mask_matrix[row])[0]
            if int(mask_positions.numel()) == 0:
                continue
            pos = int(mask_positions[0].item())
            probs = torch.softmax(outputs[row, pos, :], dim=-1)
            values, indices = torch.topk(probs, topk)
            tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
            cleaned_tokens: List[str] = []
            for tok in tokens:
                text = tokenizer.convert_tokens_to_string([tok]).strip()
                cleaned_tokens.append(text if text else tok.strip())
            per_item[item_idx] = list(zip(cleaned_tokens, values.tolist()))
    return per_item


def score_options_independent_batch(
    model,
    tokenizer: PreTrainedTokenizer,
    batch_items: List[Dict[str, object]],
    device: str,
    max_seq_length: int,
    need_vocab: bool = False,
    vocab_topk: int = 3,
) -> Tuple[List[Dict[str, float]], List[List[Tuple[str, float]]]]:
    """
    Batched scorer for independent mode.
    Each batch item must include: {"sentence": str, "options": List[str]}.
    """
    model.eval()
    scores_per_item: List[Dict[str, float]] = [{} for _ in batch_items]
    vocab_per_item: List[List[Tuple[str, float]]] = [[] for _ in batch_items]
    vocab_candidates: Dict[int, Tuple[int, List[Tuple[str, float]]]] = {}

    # Group by tokenized option length so each item-length pair needs one forward pass.
    length_payloads: Dict[int, List[Tuple[int, List[Tuple[str, List[int]]], Dict[str, List[int]]]]] = defaultdict(list)
    for idx, item in enumerate(batch_items):
        sentence = str(item["sentence"])
        options = list(item["options"])  # type: ignore[arg-type]
        option_map = build_option_map(tokenizer, sentence, options)
        if not option_map:
            continue
        grouped: Dict[int, List[Tuple[str, List[int]]]] = defaultdict(list)
        for opt_text, opt_ids in option_map:
            grouped[len(opt_ids)].append((opt_text, opt_ids))
        for length, group in grouped.items():
            enc = encode_masked_sentence(tokenizer, sentence, length, max_seq_length)
            if enc is not None:
                length_payloads[length].append((idx, group, enc))

    with torch.inference_mode():
        for length, payloads in length_payloads.items():
            model_inputs = [p[2] for p in payloads]
            inputs = tokenizer.pad(model_inputs, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            input_ids = inputs["input_ids"]
            mask_matrix = input_ids == tokenizer.mask_token_id

            for b_idx, (item_idx, group, _) in enumerate(payloads):
                mask_positions = torch.where(mask_matrix[b_idx])[0]
                if int(mask_positions.numel()) != length:
                    continue
                mask_logits = logits[b_idx, mask_positions, :]
                log_probs = torch.log_softmax(mask_logits, dim=-1)
                row_idx = torch.arange(length, device=log_probs.device)

                for opt_text, opt_ids in group:
                    target_ids = torch.tensor(opt_ids, device=log_probs.device)
                    token_log_probs = log_probs[row_idx, target_ids]
                    total_log_prob = float(token_log_probs.sum().item())
                    scores_per_item[item_idx][opt_text] = total_log_prob / length
                if need_vocab and int(mask_positions.numel()) > 0:
                    # Reuse existing logits for vocab debug output:
                    # prefer length==1 context; otherwise fallback to first available length.
                    first_mask_logits = mask_logits[0, :]
                    probs = torch.softmax(first_mask_logits, dim=-1)
                    values, indices = torch.topk(probs, vocab_topk)
                    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
                    cleaned_tokens: List[str] = []
                    for tok in tokens:
                        text = tokenizer.convert_tokens_to_string([tok]).strip()
                        cleaned_tokens.append(text if text else tok.strip())
                    priority = 0 if length == 1 else 1
                    candidate = list(zip(cleaned_tokens, values.tolist()))
                    prev = vocab_candidates.get(item_idx)
                    if prev is None or priority < prev[0]:
                        vocab_candidates[item_idx] = (priority, candidate)

        if need_vocab and not vocab_candidates:
            # Fallback only when no reusable logits were available.
            vocab_per_item = batched_vocab_predictions(
                model,
                tokenizer,
                batch_items,
                device,
                max_seq_length=max_seq_length,
                topk=vocab_topk,
            )
        elif need_vocab:
            for item_idx, (_, candidate) in vocab_candidates.items():
                vocab_per_item[item_idx] = candidate

    return scores_per_item, vocab_per_item


def score_options_pll_batch(
    model,
    tokenizer: PreTrainedTokenizer,
    batch_items: List[Dict[str, object]],
    device: str,
    max_seq_length: int,
    variant_batch_size: int = 256,
) -> List[Dict[str, float]]:
    """Batched PLL scorer over a chunk of items."""
    model.eval()
    scores_per_item: List[Dict[str, float]] = [{} for _ in batch_items]
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        return scores_per_item

    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or len(tokenizer))
    effective_variant_batch_size = cap_pll_variant_batch_size(
        requested_batch_size=variant_batch_size,
        max_seq_length=max_seq_length,
        vocab_size=vocab_size,
        device=device,
    )

    packed_variants: List[Dict[str, object]] = []
    for item_idx, item in enumerate(batch_items):
        sentence = str(item["sentence"])
        options = list(item["options"])  # type: ignore[arg-type]
        option_map = build_option_map(tokenizer, sentence, options)
        if not option_map:
            continue
        grouped: Dict[int, List[Tuple[str, List[int]]]] = defaultdict(list)
        for opt_text, opt_ids in option_map:
            grouped[len(opt_ids)].append((opt_text, opt_ids))

        for length, group in grouped.items():
            enc = encode_masked_sentence(tokenizer, sentence, length, max_seq_length)
            if enc is None:
                continue
            base_ids = enc["input_ids"]
            base_attn = enc.get("attention_mask", [1] * len(base_ids))
            base_tti = enc.get("token_type_ids")
            mask_positions = [i for i, tok_id in enumerate(base_ids) if tok_id == mask_token_id]
            if len(mask_positions) != length:
                continue

            for opt_text, opt_ids in group:
                if len(opt_ids) != length:
                    continue
                filled_ids = base_ids.copy()
                for pos, target_id in zip(mask_positions, opt_ids):
                    filled_ids[pos] = int(target_id)

                for pos, target_id in zip(mask_positions, opt_ids):
                    variant_ids = filled_ids.copy()
                    variant_ids[pos] = mask_token_id
                    row: Dict[str, object] = {
                        "input_ids": variant_ids,
                        "attention_mask": base_attn.copy(),
                        "_item_idx": item_idx,
                        "_opt_text": opt_text,
                        "_target_id": int(target_id),
                        "_mask_pos": int(pos),
                    }
                    if base_tti is not None:
                        row["token_type_ids"] = base_tti.copy()
                    packed_variants.append(row)

    if not packed_variants:
        return scores_per_item

    accum: Dict[Tuple[int, str], Tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    with torch.inference_mode():
        for start in range(0, len(packed_variants), effective_variant_batch_size):
            chunk = packed_variants[start:start + effective_variant_batch_size]
            model_inputs = [
                {k: v for k, v in row.items() if not k.startswith("_")}
                for row in chunk
            ]
            inputs = tokenizer.pad(model_inputs, return_tensors="pt", padding=True).to(device)
            logits = model(**inputs).logits
            for row_idx, row in enumerate(chunk):
                pos = int(row["_mask_pos"])
                target_id = int(row["_target_id"])
                item_idx = int(row["_item_idx"])
                opt_text = str(row["_opt_text"])
                log_prob = torch.log_softmax(logits[row_idx, pos, :], dim=-1)[target_id]
                prev_sum, prev_count = accum[(item_idx, opt_text)]
                accum[(item_idx, opt_text)] = (prev_sum + float(log_prob.item()), prev_count + 1)

    for (item_idx, opt_text), (sum_log_prob, count) in accum.items():
        if count > 0:
            scores_per_item[item_idx][opt_text] = sum_log_prob / count

    return scores_per_item


def score_options(
    model,
    tokenizer,
    sentence: str,
    options: List[str],
    device: str,
    max_seq_length: int,
    scoring_mode: str = "independent",
) -> Dict[str, float]:
    model.eval()
    option_map = build_option_map(tokenizer, sentence, options)

    if not option_map:
        return {}

    length_groups = defaultdict(list)
    for opt_text, opt_ids in option_map:
        length_groups[len(opt_ids)].append((opt_text, opt_ids))

    scores = {}

    with torch.inference_mode():
        for length, group in length_groups.items():
            enc = encode_masked_sentence(tokenizer, sentence, length, max_seq_length)
            if enc is None:
                continue
            inputs = tokenizer.pad([enc], return_tensors="pt", padding=True).to(device)
            input_ids = inputs["input_ids"]

            mask_token_id = tokenizer.mask_token_id
            mask_positions = torch.where(input_ids[0] == mask_token_id)[0]

            # Safety check: Mask count mismatch
            if int(mask_positions.numel()) != length:
                continue

            mask_positions_list = [int(x) for x in mask_positions.tolist()]

            if scoring_mode == "independent":
                outputs = model(**inputs)
                logits = outputs.logits
                mask_logits = logits[0, mask_positions, :]
                log_probs = torch.log_softmax(mask_logits, dim=-1)
                row_idx = torch.arange(length, device=log_probs.device)

                for opt_text, opt_ids in group:
                    target_ids = torch.tensor(opt_ids, device=log_probs.device)
                    token_log_probs = log_probs[row_idx, target_ids]
                    total_log_prob = token_log_probs.sum().item()
                    scores[opt_text] = total_log_prob / length
                continue

            # Pseudo-log-likelihood mode:
            # Score each option by masking one target token at a time, while
            # conditioning on the other target tokens filled in context.
            shared_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}

            for opt_text, opt_ids in group:
                if len(opt_ids) != length:
                    continue

                working_input_ids = input_ids.clone()
                target_ids = [int(tok_id) for tok_id in opt_ids]

                for pos, target_id in zip(mask_positions_list, target_ids):
                    working_input_ids[0, pos] = target_id

                # Build one batched forward per option instead of one per token.
                variant_input_ids = working_input_ids.repeat(length, 1)
                for row_idx, pos in enumerate(mask_positions_list):
                    variant_input_ids[row_idx, pos] = mask_token_id

                expanded_inputs = {}
                for key, value in shared_inputs.items():
                    repeats = (length,) + (1,) * (value.dim() - 1)
                    expanded_inputs[key] = value.repeat(*repeats)

                outputs = model(input_ids=variant_input_ids, **expanded_inputs)
                logits = outputs.logits
                row_idx = torch.arange(length, device=logits.device)
                pos_idx = torch.tensor(mask_positions_list, device=logits.device)
                mask_logits = logits[row_idx, pos_idx, :]
                log_probs = torch.log_softmax(mask_logits, dim=-1)
                target_tensor = torch.tensor(target_ids, device=logits.device)
                token_log_probs = log_probs[row_idx, target_tensor]
                total_log_prob = float(token_log_probs.sum().item())
                scores[opt_text] = total_log_prob / length

    return scores

def get_vocab_predictions(
    model,
    tokenizer,
    sentence: str,
    device: str,
    max_seq_length: int,
    topk: int = 3
) -> List[Tuple[str, float]]:
    """
    Returns the model's own top-k predictions for the masked token.
    """
    model.eval()
    enc = encode_masked_sentence(tokenizer, sentence, 1, max_seq_length)
    if enc is None:
        return []
    inputs = tokenizer.pad([enc], return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]
    mask_positions = torch.where(input_ids[0] == tokenizer.mask_token_id)[0]

    if int(mask_positions.numel()) == 0:
        return []

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits
        mask_logits = logits[0, int(mask_positions[0].item()), :]
        probs = torch.softmax(mask_logits, dim=-1)
        values, indices = torch.topk(probs, topk)

    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    cleaned_tokens = []
    for tok in tokens:
        # Try to render BPE/Subword tokens in a human-friendly way
        text = tokenizer.convert_tokens_to_string([tok]).strip()
        cleaned_tokens.append(text if text else tok.strip())

    return list(zip(cleaned_tokens, values.tolist()))

def record_verbose_entry(store, doc_id, sentence, model_name, payload):
    if store is None:
        return
    slot = store.setdefault(doc_id, {"sentence": sentence, "models": {}})
    if not slot.get("sentence"):
        slot["sentence"] = sentence
    slot["models"][model_name] = payload

def evaluate_model(
    model_name,
    dataset,
    device,
    topk,
    max_seq_length=512,
    eval_batch_size=32,
    pll_variant_batch_size=256,
    verbose=False,
    verbose_store=None,
    scoring_mode="independent",
) -> Optional[Tuple[Dict[str, Any], Dict[str, bool]]]:
    logger.info(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        vocab_expanded = ensure_padding_token(tokenizer)
        if vocab_expanded:
            model.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return None

    stats = {
        "total": 0,
        "top1": 0,
        "topk": 0,
        "domains": defaultdict(lambda: {"total": 0, "top1": 0, "topk": 0})
    }
    top1_hits: Dict[str, bool] = {}

    def process_item(item, scores: Dict[str, float], vocab_preds: List[Tuple[str, float]] | None = None):
        sentence = item["sentence"]
        options = item["options"]
        gold = item["gold"]
        domain = item.get("metadata", {}).get("law_area", "unknown")
        doc_id = item.get("_doc_id") or item.get("id") or item.get("doc_id") or f"ITEM_{stats['total']+1}"

        if gold not in options:
            if verbose:
                record_verbose_entry(
                    verbose_store,
                    doc_id,
                    sentence,
                    model_name,
                    {"status": "SKIP", "reason": "Gold not in options"},
                )
            return

        if gold not in scores:
            if verbose:
                record_verbose_entry(
                    verbose_store,
                    doc_id,
                    sentence,
                    model_name,
                    {"status": "SKIP", "reason": "Gold not scored"},
                )
            return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_opts = [r[0] for r in ranked]
        stats["total"] += 1
        stats["domains"][domain]["total"] += 1

        is_top1 = ranked_opts[0] == gold
        is_topk = gold in ranked_opts[:topk]
        top1_hits[str(doc_id)] = is_top1

        if is_top1:
            stats["top1"] += 1
            stats["domains"][domain]["top1"] += 1
        if is_topk:
            stats["topk"] += 1
            stats["domains"][domain]["topk"] += 1

        if verbose:
            max_score = max(scores.values())
            exp_scores = {opt: math.exp(score - max_score) for opt, score in scores.items()}
            denom = sum(exp_scores.values()) or 1.0
            ratios = {opt: exp_scores[opt] / denom for opt in exp_scores}
            display_order = [gold] + [opt for opt in options if opt != gold]
            record_verbose_entry(
                verbose_store,
                doc_id,
                sentence,
                model_name,
                {
                    "status": "WIN" if is_top1 else "FAIL",
                    "is_top1": is_top1,
                    "ranked_opts": ranked_opts,
                    "display_order": display_order,
                    "ratios": ratios,
                    "vocab": vocab_preds or [],
                },
            )

    if scoring_mode == "independent":
        if eval_batch_size <= 0:
            eval_batch_size = 32
        iterator = tqdm(range(0, len(dataset), eval_batch_size), desc=f"Evaluating {model_name}", unit="batch")
        for start in iterator:
            chunk = dataset[start:start + eval_batch_size]
            payload = [{"sentence": item["sentence"], "options": item["options"]} for item in chunk]
            batch_scores, batch_vocab = score_options_independent_batch(
                model,
                tokenizer,
                payload,
                device,
                max_seq_length=max_seq_length,
                need_vocab=verbose,
                vocab_topk=3,
            )
            for idx, item in enumerate(chunk):
                process_item(item, batch_scores[idx], batch_vocab[idx] if verbose else None)
    else:
        if eval_batch_size <= 0:
            eval_batch_size = 8
        vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or len(tokenizer))
        safe_pll_variant_batch_size = cap_pll_variant_batch_size(
            requested_batch_size=pll_variant_batch_size,
            max_seq_length=max_seq_length,
            vocab_size=vocab_size,
            device=device,
        )
        if safe_pll_variant_batch_size < pll_variant_batch_size:
            logger.warning(
                "Reducing pll_variant_batch_size from %d to %d for device=%s (max_seq_length=%d, vocab_size=%d).",
                pll_variant_batch_size,
                safe_pll_variant_batch_size,
                device,
                max_seq_length,
                vocab_size,
            )
        iterator = tqdm(range(0, len(dataset), eval_batch_size), desc=f"Evaluating {model_name}", unit="batch")
        for start in iterator:
            chunk = dataset[start:start + eval_batch_size]
            payload = [{"sentence": item["sentence"], "options": item["options"]} for item in chunk]
            batch_scores = score_options_pll_batch(
                model,
                tokenizer,
                payload,
                device,
                max_seq_length=max_seq_length,
                variant_batch_size=safe_pll_variant_batch_size,
            )
            batch_vocab = batched_vocab_predictions(
                model,
                tokenizer,
                payload,
                device,
                max_seq_length=max_seq_length,
                topk=3,
            ) if verbose else [[] for _ in chunk]
            for idx, item in enumerate(chunk):
                process_item(item, batch_scores[idx], batch_vocab[idx] if verbose else None)

    # Calculate final metrics
    top1_ci = wilson_interval(stats["top1"], stats["total"])
    topk_ci = wilson_interval(stats["topk"], stats["total"])
    results = {
        "model": model_name,
        "n": stats["total"],
        "top1_correct": stats["top1"],
        "topk_correct": stats["topk"],
        "acc_top1": stats["top1"] / stats["total"] if stats["total"] else 0,
        "acc_topk": stats["topk"] / stats["total"] if stats["total"] else 0,
        "acc_top1_ci95_low": top1_ci[0],
        "acc_top1_ci95_high": top1_ci[1],
        "acc_topk_ci95_low": topk_ci[0],
        "acc_topk_ci95_high": topk_ci[1],
        "scoring_mode": scoring_mode,
        "domains": {}
    }

    for d, d_stats in stats["domains"].items():
        if d_stats["total"] > 0:
            results["domains"][d] = {
                "n": d_stats["total"],
                "acc_top1": d_stats["top1"] / d_stats["total"],
                "acc_topk": d_stats["topk"] / d_stats["total"]
            }

    return (results, top1_hits)

def print_verbose_comparisons(dataset, verbose_store, model_order):
    if not verbose_store or not model_order:
        return

    separator = "-" * 27
    print(f"\n{separator}")
    for item in dataset:
        doc_id = item.get("_doc_id") or item.get("id") or item.get("doc_id")
        if not doc_id:
            continue
        slot = verbose_store.get(doc_id)
        if not slot or not slot.get("models"):
            continue
        sentence = slot.get("sentence") or item.get("sentence", "")
        print(f"[{doc_id}] {sentence}")

        for model_name in model_order:
            entry = slot["models"].get(model_name)
            if not entry:
                print(f"**{model_name}** (SKIP)")
                print("  Skipped: No evaluation result")
                continue

            status = entry.get("status", "SKIP")
            if status == "SKIP":
                print(f"**{model_name}** (SKIP)")
                reason = entry.get("reason")
                if reason:
                    print(f"  Skipped: {reason}")
                continue

            is_win = entry.get("is_top1", False)
            color = "\033[92m" if is_win else "\033[91m"
            reset = "\033[0m"
            print(f"{color}**{model_name}** ({status}){reset}")

            ranked_opts = entry.get("ranked_opts", [])
            ratios = entry.get("ratios", {})
            display_order = entry.get("display_order") or ranked_opts
            if display_order:
                opts_str = "  ".join([f"{opt}: {ratios.get(opt, 0.0):.2f}" for opt in display_order])
                print(f"  Tops: {opts_str}")

            vocab_preds = entry.get("vocab") or []
            if vocab_preds:
                vocab_str = ", ".join([f"{token}={prob:.3f}" for token, prob in vocab_preds])
                print(f"  Vocab: {vocab_str}")

        print(separator)

def print_table(results_list, topk):
    sorted_results = sorted(results_list, key=lambda x: x["acc_top1"], reverse=True)

    headers = ["Model Name", "N", "Top-1", "Top-1 95% CI", f"Top-{topk}", f"Top-{topk} 95% CI"]
    rows = [
        [
            res["model"],
            str(res["n"]),
            f"{res['acc_top1']:.2%}",
            f"[{res['acc_top1_ci95_low']:.2%}, {res['acc_top1_ci95_high']:.2%}]",
            f"{res['acc_topk']:.2%}",
            f"[{res['acc_topk_ci95_low']:.2%}, {res['acc_topk_ci95_high']:.2%}]",
        ]
        for res in sorted_results
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def row_txt(values):
        return "| " + " | ".join(
            str(v).ljust(widths[i]) if i == 0 else str(v).rjust(widths[i])
            for i, v in enumerate(values)
        ) + " |"

    print("\nModel Performance")
    print(line)
    print(row_txt(headers))
    print("+" + "+".join("-" * (w + 2) for w in widths) + "+")
    for row in rows:
        print(row_txt(row))
    print(line)
    print()


def print_mcnemar_vs_best(results_list, top1_hits_by_model):
    if len(results_list) < 2:
        return

    sorted_results = sorted(results_list, key=lambda x: x["acc_top1"], reverse=True)
    best = sorted_results[0]
    best_name = best["model"]
    best_hits = top1_hits_by_model.get(best_name, {})

    comparisons = []
    for res in sorted_results[1:]:
        other_name = res["model"]
        other_hits = top1_hits_by_model.get(other_name, {})
        cmp_result = compare_top1_with_mcnemar(best_name, other_name, best_hits, other_hits)
        if cmp_result is not None:
            comparisons.append(cmp_result)

    if not comparisons:
        return

    headers = [
        "Best Model",
        "Other Model",
        "N(common)",
        "Best only",
        "Other only",
        "Exact p",
        "Chi2-cc p",
    ]
    rows = []
    for cmp_result in comparisons:
        rows.append(
            [
                str(cmp_result["model_a"]),
                str(cmp_result["model_b"]),
                str(cmp_result["n_common"]),
                str(cmp_result["a_only"]),
                str(cmp_result["b_only"]),
                f"{cmp_result['exact_p']:.4g}",
                f"{cmp_result['chi2_cc_p']:.4g}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def row_txt(values):
        return "| " + " | ".join(
            str(v).ljust(widths[i]) if i in (0, 1) else str(v).rjust(widths[i])
            for i, v in enumerate(values)
        ) + " |"

    print("McNemar Top-1 (Best vs Others)")
    print(line)
    print(row_txt(headers))
    print("+" + "+".join("-" * (w + 2) for w in widths) + "+")
    for row in rows:
        print(row_txt(row))
    print(line)
    print("Interpretation: Exact p < 0.05 suggests a statistically significant Top-1 difference.")
    print()

def main():
    parser = argparse.ArgumentParser(description="Robust Masked Language Model Evaluator")
    parser.add_argument("--data-path", required=True, help="Path to JSONL dataset")
    parser.add_argument("--models", nargs="+", required=True, help="Model names or paths")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, auto")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length for evaluation encodings.")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Batch size for independent scoring mode.")
    parser.add_argument("--pll-variant-batch-size", type=int, default=256, help="Batch size for PLL variant forwards.")
    parser.add_argument(
        "--multiword-scoring",
        choices=["pll", "independent"],
        default="independent",
        help="Scoring strategy for masked options (default: independent).",
    )
    parser.add_argument("--output-json", help="Path to save detailed results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.max_seq_length <= 0:
        logger.error("--max-seq-length must be positive.")
        sys.exit(1)
    if args.eval_batch_size <= 0:
        logger.error("--eval-batch-size must be positive.")
        sys.exit(1)
    if args.pll_variant_batch_size <= 0:
        logger.error("--pll-variant-batch-size must be positive.")
        sys.exit(1)

    device = resolve_device(args.device)
    logger.info(f"Running on device: {device}")

    dataset = load_dataset(args.data_path)
    logger.info(f"Loaded {len(dataset)} items.")

    all_results = []
    verbose_store = {} if args.verbose else None
    evaluated_models = []
    top1_hits_by_model: Dict[str, Dict[str, bool]] = {}
    for model_name in args.models:
        eval_output = evaluate_model(
            model_name,
            dataset,
            device,
            args.topk,
            args.max_seq_length,
            args.eval_batch_size,
            args.pll_variant_batch_size,
            args.verbose,
            verbose_store=verbose_store,
            scoring_mode=args.multiword_scoring,
        )
        if eval_output:
            res, top1_hits = eval_output
            all_results.append(res)
            evaluated_models.append(model_name)
            top1_hits_by_model[model_name] = top1_hits

    if not all_results:
        logger.error("No models evaluated successfully.")
        sys.exit(1)

    if args.verbose:
        print_verbose_comparisons(dataset, verbose_store or {}, evaluated_models)

    print_table(all_results, args.topk)
    print_mcnemar_vs_best(all_results, top1_hits_by_model)

    # Detailed Domain breakdown for the best model
    best_model = max(all_results, key=lambda x: x["acc_top1"])
    print(f"Best Model Breakdown ({best_model['model']}):")
    domain_headers = ["Domain", "N", "Top-1"]
    domain_rows = [
        [domain, str(metrics["n"]), f"{metrics['acc_top1']:.2%}"]
        for domain, metrics in sorted(best_model["domains"].items())
    ]
    domain_widths = [len(h) for h in domain_headers]
    for row in domain_rows:
        for i, cell in enumerate(row):
            domain_widths[i] = max(domain_widths[i], len(cell))

    domain_line = "+" + "+".join("-" * (w + 2) for w in domain_widths) + "+"

    def domain_row_txt(values):
        return "| " + " | ".join(
            str(v).ljust(domain_widths[i]) if i == 0 else str(v).rjust(domain_widths[i])
            for i, v in enumerate(values)
        ) + " |"

    print(domain_line)
    print(domain_row_txt(domain_headers))
    print(domain_line)
    for row in domain_rows:
        print(domain_row_txt(row))
    print(domain_line)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved results to {args.output_json}")

if __name__ == "__main__":
    main()
