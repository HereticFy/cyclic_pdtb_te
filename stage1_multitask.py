"""Stage 1: Multi-task Learning (Section 3.1 of the paper).

The LLM is fine-tuned with LoRA on the TRE and the IDRP training sets jointly, using alternating-batch training
with gradient accumulation. In every training iteration the model consecutively processes a TRE batch and an
IDRP batch:

- TRE batch: prompt of Figure 2, target "Relation: <temporal relation>", loss L_te;
- IDRP batch: prompt of Figure 3, target "Connective: <connective>, Relation: <discourse relation>"
  (joint loss for the connective and the discourse relation), loss L_dr.

Back-propagation is performed for each loss within the same iteration, and the parameters are updated only
after the gradients of both L_te and L_dr have been accumulated.

The saved LoRA adapter is the starting point of Stage 2 (stage2_cross_task.py --stage1_adapter).
Run `python stage1_multitask.py --help` for all arguments; see README.md for an example.
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from common import (ChatFormatter, backward_on_batch, generate, load_data, make_batches, optimizer_step,
                    pdtb_prompt, set_seed, te_prompt, write_predictions)


def get_args():
    p = argparse.ArgumentParser(description="Stage 1: multi-task learning on TRE and IDRP")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--pdtb_type", required=True, choices=["pdtb2_l1", "pdtb2_l2", "pdtb3_l1", "pdtb3_l2"])
    p.add_argument("--te_task", required=True, choices=["matres", "tbd"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", required=True)

    # LoRA (no defaults: set them for your own setting)
    p.add_argument("--lora_rank", type=int, required=True)
    p.add_argument("--lora_alpha", type=int, required=True)
    p.add_argument("--lora_dropout", type=float, required=True)
    p.add_argument("--lora_target_modules", nargs="+", required=True,
                   help="e.g. q_proj k_proj v_proj o_proj gate_proj up_proj down_proj")

    # Fine-tuning hyper-parameters (no defaults)
    p.add_argument("--num_train_epochs", type=float, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, required=True)
    p.add_argument("--gradient_accumulation_steps", type=int, required=True,
                   help="Micro-batches per task batch.")
    p.add_argument("--learning_rate", type=float, required=True)
    p.add_argument("--weight_decay", type=float, required=True)
    p.add_argument("--lr_scheduler_type", required=True, help="e.g. cosine, linear, constant (transformers.get_scheduler).")
    p.add_argument("--warmup_ratio", type=float, required=True)
    p.add_argument("--max_grad_norm", type=float, required=True)
    p.add_argument("--cutoff_len", type=int, required=True)
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Prediction on the test sets (required with --do_predict)
    p.add_argument("--do_predict", action="store_true", help="Predict on the test sets after training.")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--eval_max_new_tokens", type=int)
    p.add_argument("--eval_temperature", type=float)
    p.add_argument("--eval_top_p", type=float)

    p.add_argument("--max_samples", type=int, default=None, help="Limit samples per split (for debugging).")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=0, help="Save the adapter every N iterations (0: only at the end).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.do_predict and None in (args.eval_max_new_tokens, args.eval_temperature, args.eval_top_p):
        p.error("--do_predict requires --eval_max_new_tokens, --eval_temperature and --eval_top_p")
    return args


def train(model, fmt, te_train, pdtb_train, args):
    task_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    rng = random.Random(args.seed)
    iters_per_epoch = max(math.ceil(len(te_train) / task_batch_size), math.ceil(len(pdtb_train) / task_batch_size))
    total_iters = int(iters_per_epoch * args.num_train_epochs)  # one parameter update per iteration

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_scheduler(args.lr_scheduler_type, optimizer,
                              num_warmup_steps=int(total_iters * args.warmup_ratio), num_training_steps=total_iters)

    te_batches, pdtb_batches, log = [], [], []
    for it in tqdm(range(total_iters), desc="Stage 1 iterations"):
        if not te_batches:
            te_batches = make_batches(te_train, task_batch_size, rng)
        if not pdtb_batches:
            pdtb_batches = make_batches(pdtb_train, task_batch_size, rng)
        te_batch, pdtb_batch = te_batches.pop(), pdtb_batches.pop()

        # TRE batch, then IDRP batch; both losses are back-propagated before the single update
        loss_te = backward_on_batch(model, fmt, [te_prompt(s, args.te_task) for s in te_batch],
                                    args.per_device_train_batch_size)
        loss_dr = backward_on_batch(model, fmt, [pdtb_prompt(s, args.pdtb_type) for s in pdtb_batch],
                                    args.per_device_train_batch_size)
        optimizer_step(model, optimizer, scheduler, args.max_grad_norm)

        log.append({"iteration": it + 1, "loss_te": loss_te, "loss_dr": loss_dr, "lr": scheduler.get_last_lr()[0]})
        if (it + 1) % args.logging_steps == 0:
            recent = log[-args.logging_steps:]
            print(f"iter {it + 1}/{total_iters} | loss_te {np.mean([r['loss_te'] for r in recent]):.4f} "
                  f"| loss_dr {np.mean([r['loss_dr'] for r in recent]):.4f} | lr {log[-1]['lr']:.2e}")
        if args.save_steps and (it + 1) % args.save_steps == 0:
            model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{it + 1}"))

    with open(os.path.join(args.output_dir, "train_log.jsonl"), "w", encoding="utf-8") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")


def predict(model, fmt, pairs, args, out_dir):
    preds = generate(model, fmt, [p for p, _ in pairs], args.eval_batch_size,
                     args.eval_max_new_tokens, args.eval_temperature, args.eval_top_p, desc="Prediction")
    rows, correct = [], 0
    for (prompt, label), pred in zip(pairs, preds):
        rows.append({"prompt": prompt, "predict": pred, "label": label})
        correct += pred.lower().split("relation:")[-1].strip() == label.lower().split("relation:")[-1].strip()
    write_predictions(os.path.join(out_dir, "generated_predictions.jsonl"), rows)
    print(f"{out_dir}: exact-match accuracy {correct / max(len(pairs), 1):.4f} (use eval.py for the paper metrics)")


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
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.lora_target_modules))
    model.print_trainable_parameters()

    te_train, pdtb_train = load_data(args.data_dir, args.te_task, args.pdtb_type, "train", args.max_samples)
    print(f"TRE train: {len(te_train)} | IDRP train: {len(pdtb_train)}")

    train(model, fmt, te_train, pdtb_train, args)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.do_predict:
        te_test, pdtb_test = load_data(args.data_dir, args.te_task, args.pdtb_type, "test", args.max_samples)
        predict(model, fmt, [te_prompt(s, args.te_task) for s in te_test], args,
                os.path.join(args.output_dir, f"{args.te_task}_test"))
        predict(model, fmt, [pdtb_prompt(s, args.pdtb_type) for s in pdtb_test], args,
                os.path.join(args.output_dir, f"{args.pdtb_type}_test"))


if __name__ == "__main__":
    main()
