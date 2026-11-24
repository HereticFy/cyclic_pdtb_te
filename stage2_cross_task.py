"""Stage 2: Iterative Cross-task Learning.

For every training iteration (see Section 3.2 of the paper):

1. Discourse-informed temporal enhancement (TRE batch)
   a. Auxiliary inference (frozen): the model predicts a connective and a discourse relation
      between the two event-containing arguments.
   b. Augmented training: the prediction is appended to the TRE input and the model is
      fine-tuned to predict the gold temporal relation; the parameters are updated.
2. Temporal-informed discourse enhancement (IDRP batch), using the model updated in step 1
   a. Auxiliary inference (frozen): the model extracts event pairs across Arg1/Arg2 and predicts
      their temporal relations as "(Event A, Event B, Relation)" tuples.
   b. Augmented training: the tuples are appended to the IDRP input and the model is fine-tuned
      to predict the gold connective and discourse relation; the parameters are updated.

Training starts from the Stage-1 (multi-task learning) LoRA adapter. After training, the test
sets are augmented with auxiliary predictions of the final model in the same way and predicted.
Predictions are written in LLaMA-Factory's `generated_predictions.jsonl` format, so they can be
scored with eval.py.

Run `python stage2_cross_task.py --help` for all arguments; see README.md for an example.
"""
import argparse
import json
import math
import os
import random
import re
import textwrap

import numpy as np
import torch
from peft import LoraConfig, PeftModel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from common import (ChatFormatter, backward_on_batch, clean_generation, discourse_prompt, generate, label_list,
                    load_data, make_batches, optimizer_step, pdtb_label_set, set_seed, te_label_info,
                    write_predictions)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Stage 2: iterative cross-task learning")
    p.add_argument("--model_name_or_path", required=True, help="Base model used in Stage 1.")
    p.add_argument("--stage1_adapter", required=True,
                   help="LoRA adapter trained in Stage 1 (multi-task learning). Stage 2 continues training this "
                        "adapter; its LoRA configuration (rank, alpha, target modules) is reused. To run a further "
                        "round, pass the output directory of a previous Stage-2 run.")
    p.add_argument("--pdtb_type", required=True, choices=["pdtb2_l1", "pdtb2_l2", "pdtb3_l1", "pdtb3_l2"])
    p.add_argument("--te_task", required=True, choices=["matres", "tbd"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", required=True)

    # Fine-tuning hyper-parameters (no defaults: set them for your own setting)
    p.add_argument("--num_train_epochs", type=float, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, required=True)
    p.add_argument("--gradient_accumulation_steps", type=int, required=True,
                   help="Micro-batches per task batch; the parameters are updated once per task batch.")
    p.add_argument("--learning_rate", type=float, required=True)
    p.add_argument("--weight_decay", type=float, required=True)
    p.add_argument("--lr_scheduler_type", required=True, help="e.g. cosine, linear, constant (transformers.get_scheduler).")
    p.add_argument("--warmup_ratio", type=float, required=True)
    p.add_argument("--max_grad_norm", type=float, required=True)
    p.add_argument("--cutoff_len", type=int, required=True)
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Generation (auxiliary inference and final prediction)
    p.add_argument("--aux_batch_size", type=int, default=8, help="Batch size for (frozen) auxiliary inference.")
    p.add_argument("--aux_max_new_tokens", type=int, required=True)
    p.add_argument("--aux_temperature", type=float, required=True)
    p.add_argument("--aux_top_p", type=float, required=True)
    p.add_argument("--aux_followups", type=int, default=1,
                   help="Level 1: follow-up questions in the same conversation when the auxiliary answer "
                        "contains no valid prediction (e.g. no concrete event pair).")
    p.add_argument("--aux_step_attempts", type=int, default=3,
                   help="Level 2: how often each step-by-step question (core event of Arg1 / Arg2, relation; "
                        "or connective, relation) is asked until a valid answer is given.")
    p.add_argument("--aux_fallback", default="None",
                   help="Auxiliary feature used only if all guidance levels fail.")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--eval_max_new_tokens", type=int, required=True)
    p.add_argument("--eval_temperature", type=float, required=True)
    p.add_argument("--eval_top_p", type=float, required=True)

    p.add_argument("--max_samples", type=int, default=None, help="Limit samples per split (for debugging).")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=0, help="Save the adapter every N iterations (0: only at the end).")
    p.add_argument("--do_predict", action="store_true", help="Predict on the test sets after training.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Prompts of Stage 2 (paper, Figures 4-7) and follow-up questions
# ---------------------------------------------------------------------------

def te_aux_prompt(s, pdtb_type):
    """Auxiliary inference on a TRE sample: predict connective + discourse relation (paper, Figure 4)."""
    return discourse_prompt(s["arg1"], s["arg2"], pdtb_type)


def te_train_prompt(s, te_task, aux):
    """TRE input augmented with the predicted connective and discourse relation (paper, Figure 6)."""
    label_inst, _ = te_label_info(te_task)
    prompt = textwrap.dedent(f"""
    Your task is to analyze the text of Arg1 and Arg2 and then determine the temporal relation between Event {{e1}} and Event {{e2}}. You will be provided a predicted connective and a discourse relation between Arg1 and Arg2 to help you predict the temporal relation between Event {{e1}} and Event {{e2}}. You can only refer to it, but not fully trust this prediction. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
    The output format should be:
    Relation: [Your predicted relation]
    Input text:
    Arg1: {{arg1}}
    Arg2: {{arg2}}
    Predicted Connective and Discourse Relation: {{aux}}""").strip().format(
        e1=s["tagged_e1"], e2=s["tagged_e2"], arg1=s["arg1"], arg2=s["arg2"], aux=aux)
    return prompt, f"Relation: {s['label']}"


def pdtb_aux_prompt(s, te_task):
    """Auxiliary inference on an IDRP sample: extract event pairs and their temporal relations (paper, Figure 5)."""
    _, te_info = te_label_info(te_task)
    return textwrap.dedent(f"""
    Your task is to analyze the text of Arg1 and Arg2 and identify the core events (mainly verb, time expression, or noun phrase). Then determine the temporal relation between core events. The Result MUST be EXACTLY one of: {te_info}. Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
    The output MUST follow the format:
    (Event 1, Event 2, Your predicted relation)
    **Example:**
    Input:
    Arg1: He was brought to the company in September 1988 after 15 years at First Interstate Bancorp.
    Arg2: The bank had been suffering in late 1987 from a slew of bad real estate loans.
    Output:
    (suffering, brought, before); (late 1987, September 1988, before)
    Input:
    Arg1: It's now a very good time politically to get this over and done with,
    Arg2: The government, already buffeted by high interest rates and a slowing economy, has been badly hurt by last week's shake-up in Mrs. Thatcher's cabinet.
    Output:
    (get, buffeted, after); (has been badly hurt, get, before)
    Input text:
    Arg1: {{arg1}}
    Arg2: {{arg2}}""").strip().format(arg1=s["arg1"], arg2=s["arg2"])


def pdtb_train_prompt(s, pdtb_type, aux):
    """IDRP input augmented with the predicted temporal relations (paper, Figure 7)."""
    prompt = textwrap.dedent(f"""
    Your task is to analyze the discourse relation between Arg1 and Arg2. First, identify a potential connective that fits the gap. Then, determine the discourse relation label. You will be provided with the predicted temporal relations between events in Arg1 and Arg2 to help you predict the discourse relation. You can only refer to it, but not fully trust this prediction.
    Output your reasoning about the potential connective and the discourse relation. The relation MUST be EXACTLY one of: {pdtb_label_set(pdtb_type)}.
    The output format should be:
    Connective: [Your predicted connective], Relation: [Your predicted relation]
    Input text:
    Arg1: {{arg1}}
    Arg2: {{arg2}}
    Predicted temporal relation: {{aux}}""").strip().format(arg1=s["arg1"], arg2=s["arg2"], aux=aux)
    return prompt, f"Connective: {s['conn']}, Relation: {s['label']}"


def te_followup_prompt(s, te_task):
    """Level 1 (temporal): asked in the same conversation when the answer contains no concrete event pair."""
    _, te_info = te_label_info(te_task)
    return textwrap.dedent(f"""
    Your answer does not contain concrete events from the input text. Copy the exact words of the core event in Arg1 (Event 1) and of the core event in Arg2 (Event 2) from the input text, and determine the temporal relation between them. Do not write placeholders such as "Event 1". Possible labels: {te_info}.
    Arg1: {s['arg1']}
    Arg2: {s['arg2']}
    Answer only in the format:
    (Event 1, Event 2, RELATION)""").strip()


def dr_followup_prompt(s, pdtb_type):
    """Level 1 (discourse): asked in the same conversation when the answer has no valid connective / relation."""
    return textwrap.dedent(f"""
    Your answer is not in the required format. Identify a concrete connective that fits the gap between Arg1 and Arg2 and the discourse relation between them. The relation MUST be EXACTLY one of: {pdtb_label_set(pdtb_type)}.
    Arg1: {s['arg1']}
    Arg2: {s['arg2']}
    Answer only in the format:
    Connective: [Your predicted connective], Relation: [Your predicted relation]""").strip()


def event_step_prompt(s, which):
    """Level 2 (temporal, step-by-step): ask for the core event of one argument only."""
    arg = s["arg1"] if which == 1 else s["arg2"]
    return textwrap.dedent(f"""
    Arg{which}: {arg}
    Which word or short phrase in Arg{which} expresses its core event (mainly a verb, time expression or noun phrase)? Copy it exactly from Arg{which} and answer with the copied words only.""").strip()


def te_relation_step_prompt(s, te_task, e1, e2):
    """Level 2 (temporal, step-by-step): ask for the relation between the two given events."""
    _, te_info = te_label_info(te_task)
    return textwrap.dedent(f"""
    Arg1: {s['arg1']}
    Arg2: {s['arg2']}
    What is the temporal relation between the event "{e1}" in Arg1 and the event "{e2}" in Arg2? Answer with exactly one of: {te_info}.""").strip()


def connective_step_prompt(s):
    """Level 2 (discourse, step-by-step): ask for the connective only."""
    return textwrap.dedent(f"""
    Arg1: {s['arg1']}
    Arg2: {s['arg2']}
    Which connective (for example: because, but, and, then, for example) best fits the gap between Arg1 and Arg2? Answer with the connective only.""").strip()


def dr_relation_step_prompt(s, pdtb_type, conn):
    """Level 2 (discourse, step-by-step): ask for the relation given the connective."""
    return textwrap.dedent(f"""
    Arg1: {s['arg1']}
    Arg2: {s['arg2']}
    Connective: {conn}
    What is the discourse relation between Arg1 and Arg2? Answer with exactly one of: {pdtb_label_set(pdtb_type)}.""").strip()


PLACEHOLDER = re.compile(r"^(event\s*\d+|event\s*[ab]|e\d+|relation|connective|\[.*\]|your predicted .*|none|n/?a|)$", re.IGNORECASE)


def _strip_tags(text):
    return re.sub(r"</?[a-zA-Z]+\d*>", "", text).lower()


def occurs_in(phrase, text):
    """Whole-word, case-insensitive occurrence of `phrase` in `text` (event tags in `text` are ignored)."""
    phrase = phrase.strip().lower()
    return bool(phrase) and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", _strip_tags(text)) is not None


def _short_answer(text):
    """First line of an answer without quotes, labels such as 'Event:' and trailing punctuation."""
    line = clean_generation(text).split("\n")[0].strip()
    line = re.sub(r"^(answer|event\s*\d*|core event|connective|relation)\s*:\s*", "", line, flags=re.IGNORECASE)
    return line.strip().strip("\"'`*").rstrip(".").strip()


def match_label(text, labels):
    """Return the label (original casing) named in a short answer, or None if none / ambiguous."""
    ans = _short_answer(text).lower()
    by_lower = {l.lower(): l for l in labels}
    if ans in by_lower:
        return by_lower[ans]
    found = [l for l in labels if re.search(r"(?<![\w+-])" + re.escape(l.lower()) + r"(?![\w+-])", ans)]
    return found[0] if len(found) == 1 else None


def parse_event(text, arg):
    """A concrete event copied from the argument (at most 8 words), or None."""
    ev = _short_answer(text)
    if PLACEHOLDER.match(ev) or len(ev.split()) > 8 or not occurs_in(ev, arg):
        return None
    return ev


def parse_connective(text):
    conn = _short_answer(text)
    if PLACEHOLDER.match(conn) or len(conn.split()) > 4 or not re.search(r"[a-zA-Z]", conn):
        return None
    return conn


def parse_te_aux(text, s, te_task):
    """Keep only grounded '(Event A, Event B, Relation)' tuples: both events are concrete words that occur
    in different arguments and the relation is a valid label. Returns None if no tuple is left."""
    allowed = {l.lower() for l in label_list(te_label_info(te_task)[1])}
    kept = []
    for m in re.findall(r"\(([^()]+)\)", clean_generation(text)):
        parts = [p.strip().strip("'\"") for p in m.split(",")]
        if len(parts) != 3:
            continue
        e1, e2, rel = parts[0], parts[1], parts[2].lower()
        if PLACEHOLDER.match(e1) or PLACEHOLDER.match(e2) or rel not in allowed:
            continue
        if not ((occurs_in(e1, s["arg1"]) and occurs_in(e2, s["arg2"])) or
                (occurs_in(e1, s["arg2"]) and occurs_in(e2, s["arg1"]))):
            continue
        t = f"({e1}, {e2}, {rel})"
        if t not in kept:
            kept.append(t)
    return "; ".join(kept) if kept else None


def parse_dr_aux(text, pdtb_type):
    """Return 'Connective: X, Relation: Y' if both are present and Y is a valid label, else None."""
    text = clean_generation(text)
    conn = re.search(r"Connective:\s*([^,\n]+)", text, re.IGNORECASE)
    rel = re.search(r"Relation:\s*([^,.\n]+)", text, re.IGNORECASE)
    if not conn or not rel:
        return None
    conn, rel = conn.group(1).strip(), rel.group(1).strip()
    labels = {l.lower(): l for l in label_list(pdtb_label_set(pdtb_type))}
    if PLACEHOLDER.match(conn) or rel.lower() not in labels:
        return None
    return f"Connective: {conn}, Relation: {labels[rel.lower()]}"


# ---------------------------------------------------------------------------
# Auxiliary inference with increasingly specific guidance
# ---------------------------------------------------------------------------

GUIDANCE_LEVELS = ["first answer", "follow-up", "step-by-step", "fallback"]


def ask_until_valid(model, fmt, idx, prompt_fn, parse_fn, args):
    """Ask `prompt_fn(i)` for every sample index in `idx` and re-ask the failed ones (new samples drawn) up to
    `args.aux_step_attempts` times in total. Returns {i: parsed value} for the samples that succeeded."""
    values, pending = {}, list(idx)
    for _ in range(args.aux_step_attempts):
        if not pending:
            break
        outs = generate(model, fmt, [prompt_fn(i) for i in pending], args.aux_batch_size,
                        args.aux_max_new_tokens, args.aux_temperature, args.aux_top_p)
        still = []
        for i, out in zip(pending, outs):
            v = parse_fn(out, i)
            if v is None:
                still.append(i)
            else:
                values[i] = v
        pending = still
    return values


def auxiliary_inference(model, fmt, samples, spec, args, desc=None):
    """Frozen auxiliary inference. Guidance becomes more specific whenever no valid prediction is extracted:

    level 0  the original auxiliary prompt;
    level 1  follow-up questions in the same conversation (`args.aux_followups` times);
    level 2  the task is split into single questions (e.g. core event of Arg1 -> core event of Arg2 ->
             relation), each re-asked up to `args.aux_step_attempts` times until a valid answer is given;
    level 3  only if all of the above fail: `args.aux_fallback`.

    Returns (auxiliary features, guidance level per sample).
    """
    n = len(samples)
    results, levels = [None] * n, [0] * n

    # Levels 0 and 1: original prompt and follow-ups in the same conversation
    conversations = [[spec["first"](s)] for s in samples]
    pending = list(range(n))
    for attempt in range(args.aux_followups + 1):
        if not pending:
            break
        outs = generate(model, fmt, [conversations[i] for i in pending], args.aux_batch_size,
                        args.aux_max_new_tokens, args.aux_temperature, args.aux_top_p,
                        desc=f"{desc} (level {min(attempt, 1)})" if desc else None)
        still = []
        for i, out in zip(pending, outs):
            parsed = spec["parse"](out, samples[i])
            if parsed is not None:
                results[i], levels[i] = parsed, min(attempt, 1)
            else:
                conversations[i] += [clean_generation(out), spec["followup"](samples[i])]
                still.append(i)
        pending = still

    # Level 2: step-by-step questions; each step only for the samples that passed the previous steps
    if pending:
        partial = {i: {} for i in pending}
        alive = list(pending)
        for name, prompt_fn, parse_fn in spec["steps"]:
            got = ask_until_valid(model, fmt, alive,
                                  lambda i, f=prompt_fn: f(samples[i], partial[i]),
                                  lambda out, i, f=parse_fn: f(out, samples[i], partial[i]),
                                  args)
            for i, v in got.items():
                partial[i][name] = v
            alive = [i for i in alive if i in got]
        for i in alive:
            results[i], levels[i] = spec["compose"](partial[i]), 2
        pending = [i for i in pending if i not in set(alive)]

    # Level 3: fallback
    for i in pending:
        results[i], levels[i] = args.aux_fallback, 3
    return results, levels


def temporal_aux_spec(args):
    """Auxiliary temporal inference on IDRP samples (temporal-informed discourse enhancement)."""
    labels = label_list(te_label_info(args.te_task)[1])
    return {
        "first": lambda s: pdtb_aux_prompt(s, args.te_task),
        "parse": lambda out, s: parse_te_aux(out, s, args.te_task),
        "followup": lambda s: te_followup_prompt(s, args.te_task),
        "steps": [
            ("e1", lambda s, p: event_step_prompt(s, 1), lambda out, s, p: parse_event(out, s["arg1"])),
            ("e2", lambda s, p: event_step_prompt(s, 2), lambda out, s, p: parse_event(out, s["arg2"])),
            ("rel", lambda s, p: te_relation_step_prompt(s, args.te_task, p["e1"], p["e2"]),
             lambda out, s, p: match_label(out, labels)),
        ],
        "compose": lambda p: f"({p['e1']}, {p['e2']}, {p['rel'].lower()})",
    }


def discourse_aux_spec(args):
    """Auxiliary discourse inference on TRE samples (discourse-informed temporal enhancement)."""
    labels = label_list(pdtb_label_set(args.pdtb_type))
    return {
        "first": lambda s: te_aux_prompt(s, args.pdtb_type),
        "parse": lambda out, s: parse_dr_aux(out, args.pdtb_type),
        "followup": lambda s: dr_followup_prompt(s, args.pdtb_type),
        "steps": [
            ("conn", lambda s, p: connective_step_prompt(s), lambda out, s, p: parse_connective(out)),
            ("rel", lambda s, p: dr_relation_step_prompt(s, args.pdtb_type, p["conn"]),
             lambda out, s, p: match_label(out, labels)),
        ],
        "compose": lambda p: f"Connective: {p['conn']}, Relation: {p['rel']}",
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, fmt, te_train, pdtb_train, args):
    task_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    rng = random.Random(args.seed)
    iters_per_epoch = max(math.ceil(len(te_train) / task_batch_size), math.ceil(len(pdtb_train) / task_batch_size))
    total_iters = int(iters_per_epoch * args.num_train_epochs)
    total_updates = 2 * total_iters  # one update per TRE batch and one per IDRP batch

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_scheduler(args.lr_scheduler_type, optimizer,
                              num_warmup_steps=int(total_updates * args.warmup_ratio), num_training_steps=total_updates)

    te_batches, pdtb_batches = [], []
    log = []
    for it in tqdm(range(total_iters), desc="Stage 2 iterations"):
        if not te_batches:
            te_batches = make_batches(te_train, task_batch_size, rng)
        if not pdtb_batches:
            pdtb_batches = make_batches(pdtb_train, task_batch_size, rng)
        te_batch, pdtb_batch = te_batches.pop(), pdtb_batches.pop()

        # 1. Discourse-informed temporal enhancement (TRE batch)
        te_aux, te_levels = auxiliary_inference(model, fmt, te_batch, discourse_aux_spec(args), args)
        examples = [te_train_prompt(s, args.te_task, a) for s, a in zip(te_batch, te_aux)]
        loss_te = backward_on_batch(model, fmt, examples, args.per_device_train_batch_size)
        optimizer_step(model, optimizer, scheduler, args.max_grad_norm)

        # 2. Temporal-informed discourse enhancement (IDRP batch), with the updated model
        dr_aux, dr_levels = auxiliary_inference(model, fmt, pdtb_batch, temporal_aux_spec(args), args)
        examples = [pdtb_train_prompt(s, args.pdtb_type, a) for s, a in zip(pdtb_batch, dr_aux)]
        loss_dr = backward_on_batch(model, fmt, examples, args.per_device_train_batch_size)
        optimizer_step(model, optimizer, scheduler, args.max_grad_norm)

        log.append({"iteration": it + 1, "loss_te": loss_te, "loss_dr": loss_dr, "lr": scheduler.get_last_lr()[0],
                    "te_guidance": [te_levels.count(l) for l in range(4)],
                    "dr_guidance": [dr_levels.count(l) for l in range(4)]})
        if (it + 1) % args.logging_steps == 0:
            recent = log[-args.logging_steps:]
            print(f"iter {it + 1}/{total_iters} | loss_te {np.mean([r['loss_te'] for r in recent]):.4f} "
                  f"| loss_dr {np.mean([r['loss_dr'] for r in recent]):.4f} | lr {log[-1]['lr']:.2e} "
                  f"| guidance levels 0/1/2/3 on TRE {np.sum([r['te_guidance'] for r in recent], 0).tolist()} "
                  f"IDRP {np.sum([r['dr_guidance'] for r in recent], 0).tolist()}")
        if args.save_steps and (it + 1) % args.save_steps == 0:
            model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{it + 1}"))

    with open(os.path.join(args.output_dir, "train_log.jsonl"), "w", encoding="utf-8") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Prediction on the test sets
# ---------------------------------------------------------------------------

def predict(model, fmt, samples, aux_spec, train_prompt_fn, args, out_dir):
    """Auxiliary inference + final prediction; writes LLaMA-Factory-style generated_predictions.jsonl."""
    aux, levels = auxiliary_inference(model, fmt, samples, aux_spec, args, desc="Auxiliary inference")
    pairs = [train_prompt_fn(s, a) for s, a in zip(samples, aux)]
    preds = generate(model, fmt, [p for p, _ in pairs], args.eval_batch_size,
                     args.eval_max_new_tokens, args.eval_temperature, args.eval_top_p, desc="Prediction")

    rows, correct = [], 0
    for (prompt, label), a, lv, pred in zip(pairs, aux, levels, preds):
        rows.append({"prompt": prompt, "auxiliary": a, "guidance": GUIDANCE_LEVELS[lv], "predict": pred, "label": label})
        correct += pred.lower().split("relation:")[-1].strip() == label.lower().split("relation:")[-1].strip()
    write_predictions(os.path.join(out_dir, "generated_predictions.jsonl"), rows)
    n = max(len(samples), 1)
    dist = ", ".join(f"{name} {levels.count(l)}" for l, name in enumerate(GUIDANCE_LEVELS))
    print(f"{out_dir}: exact-match accuracy {correct / n:.4f} | auxiliary guidance: {dist} "
          f"(use eval.py for the paper metrics)")


# ---------------------------------------------------------------------------

def load_stage1_adapter(model, path):
    """Load a trainable LoRA adapter; drop config keys unknown to the installed PEFT version
    (e.g. adapters saved by a newer PEFT through LLaMA-Factory)."""
    try:
        return PeftModel.from_pretrained(model, path, is_trainable=True)
    except TypeError:
        import dataclasses
        with open(os.path.join(path, "adapter_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        known = {f.name for f in dataclasses.fields(LoraConfig)}
        dropped = sorted(k for k in cfg if k not in known)
        print(f"Ignoring adapter config keys unknown to this PEFT version: {dropped}")
        lora_config = LoraConfig(**{k: v for k, v in cfg.items() if k in known})
        return PeftModel.from_pretrained(model, path, config=lora_config, is_trainable=True)


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    fmt = ChatFormatter(tokenizer, args.cutoff_len)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = load_stage1_adapter(model, args.stage1_adapter)
    model.print_trainable_parameters()

    te_train, pdtb_train = load_data(args.data_dir, args.te_task, args.pdtb_type, "train", args.max_samples)
    print(f"TRE train: {len(te_train)} | IDRP train: {len(pdtb_train)}")

    train(model, fmt, te_train, pdtb_train, args)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.do_predict:
        te_test, pdtb_test = load_data(args.data_dir, args.te_task, args.pdtb_type, "test", args.max_samples)
        predict(model, fmt, te_test, discourse_aux_spec(args),
                lambda s, a: te_train_prompt(s, args.te_task, a),
                args, os.path.join(args.output_dir, f"{args.te_task}_test"))
        predict(model, fmt, pdtb_test, temporal_aux_spec(args),
                lambda s, a: pdtb_train_prompt(s, args.pdtb_type, a),
                args, os.path.join(args.output_dir, f"{args.pdtb_type}_test"))


if __name__ == "__main__":
    main()
