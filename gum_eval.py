"""Evaluation on eng.pdtb.gum (DISRPT 2025), following the official DISRPT relation classification scorer
(https://github.com/disrpt/sharedtask2025, utils/disrpt_eval_2024.py, task R).

The official scorer compares the last column (`label`) of the gold `.rels` file with the same column of a
prediction `.rels` file and reports the accuracy over all relation instances of the test set. This script

1. writes the predictions as a DISRPT `.rels` file (a copy of the gold file whose `label` column is replaced
   by the predicted relation), which can be passed to the official scorer:
       python utils/disrpt_eval_2024.py -g <gold .rels> -p <prediction .rels> -t R
2. computes the same accuracy (plus the official per-label classification report) directly.

The prediction file must contain exactly one prediction per relation instance of the gold `.rels` file, in the
same order (as produced from data/disrpt_data/eng.pdtb.gum_test.json).

Optionally, TRE predictions (e.g. on MATRES) can be scored with --te_predictions / --te_task.
"""
import argparse
import json
import os
import re

from sklearn.metrics import accuracy_score, classification_report

from eval import calculate_metrics_classification
from paths import DISRPT_TEST_DIR

RELS_HEADER = ("doc\tunit1_toks\tunit2_toks\tunit1_txt\tunit2_txt\tu1_raw\tu2_raw\ts1_toks\ts2_toks\t"
               "unit1_sent\tunit2_sent\tdir\trel_type\torig_label\tlabel")


def extract_relation(text):
    """Text after 'Relation:' (lower-cased, without surrounding whitespace / trailing period); '' if absent."""
    if not text:
        return ""
    match = re.search(r"Relation:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip().split("\n")[0].strip().strip(".").strip().lower()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_rels(path):
    """Header and rows of a DISRPT .rels file, read as the official scorer does."""
    data = open(path, encoding="utf-8").read().strip().replace("\r", "")
    lines = data.split("\n")
    assert lines[0] == RELS_HEADER, f"Unrecognized .rels header in {path}"
    return lines[0], lines[1:]


def write_prediction_rels(gold_rels, predictions, out_path):
    """Copy the gold .rels file and replace its `label` column with the predicted relations."""
    header, rows = read_rels(gold_rels)
    assert len(rows) == len(predictions), \
        f"{len(predictions)} predictions for {len(rows)} relation instances in {gold_rels}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row, pred in zip(rows, predictions):
            cols = row.split("\t")
            cols[-1] = pred
            f.write("\t".join(cols) + "\n")
    return out_path


def disrpt_relation_scores(gold_rels, pred_rels):
    """Accuracy and per-label report on the `label` column, as in the official scorer (task R)."""
    _, gold_rows = read_rels(gold_rels)
    _, pred_rows = read_rels(pred_rels)
    assert len(gold_rows) == len(pred_rows), "gold and prediction files have different numbers of instances"
    gold_units = [" ".join(r.split("\t")[:3]) for r in gold_rows]
    pred_units = [" ".join(r.split("\t")[:3]) for r in pred_rows]
    assert gold_units == pred_units, "gold and prediction instances are not identical"
    gold = [r.split("\t")[-1] for r in gold_rows]
    pred = [r.split("\t")[-1] for r in pred_rows]
    acc = accuracy_score(gold, pred)
    report = classification_report(gold, pred, labels=sorted(set(gold)), zero_division=0.0, digits=4)
    return acc, report


def main():
    p = argparse.ArgumentParser(description="DISRPT-style evaluation on eng.pdtb.gum")
    p.add_argument("--gum_predictions", required=True,
                   help="generated_predictions.jsonl with one prediction per relation of the gold .rels file")
    p.add_argument("--gold_rels", default=os.path.join(DISRPT_TEST_DIR, "eng.pdtb.gum", "eng.pdtb.gum_test.rels"))
    p.add_argument("--output_rels", default=None,
                   help="Where to write the prediction .rels file (default: next to --gum_predictions)")
    p.add_argument("--te_predictions", default=None, help="Optional TRE generated_predictions.jsonl")
    p.add_argument("--te_task", choices=["matres", "tbd"], default=None)
    args = p.parse_args()

    preds = [extract_relation(d["predict"]) for d in read_jsonl(args.gum_predictions)]
    out = args.output_rels or os.path.join(os.path.dirname(os.path.abspath(args.gum_predictions)),
                                           "eng.pdtb.gum_test.pred.rels")
    write_prediction_rels(args.gold_rels, preds, out)
    acc, report = disrpt_relation_scores(args.gold_rels, out)
    print(f"Prediction .rels file: {out}")
    print(report)
    print(f"eng.pdtb.gum accuracy (DISRPT): {acc * 100:.2f}")

    if args.te_predictions:
        assert args.te_task, "--te_task is required with --te_predictions"
        rows = read_jsonl(args.te_predictions)
        results = calculate_metrics_classification([extract_relation(d["predict"]) for d in rows],
                                                   [extract_relation(d["label"]) for d in rows], args.te_task)
        print(f"{args.te_task} micro-averaged scores:", results["micro_avg"])


if __name__ == "__main__":
    main()
