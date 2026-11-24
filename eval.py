import os
import json
import regex
import re
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from utils import write_to_file_json, read_from_file_json, get_list_dict_for_task
from paths import LLAMA_FACTORY_DIR, PDTB_SOURCE_DIR

def cal_acc_f1_score_with_ids(pred_ids, label_ids, possible_label_ids):
    """
    possible_label_ids: some examples may be annotated with multiple relations
    """
    label_size = possible_label_ids.shape[1]
    extend_label_ids = []
    for idx, p in enumerate(pred_ids):
        if 0 <= p < label_size and possible_label_ids[idx, p] == 1:
            extend_label_ids.append(p)
        else:
            extend_label_ids.append(label_ids[idx])
    label_ids = np.array(extend_label_ids)
    acc = accuracy_score(y_true=label_ids, y_pred=pred_ids)
    f1 = f1_score(y_true=label_ids, y_pred=pred_ids, labels=list(range(label_size)), average="macro", zero_division=0)

    return acc, f1

def cal_acc_f1_score_with_ids_v2(pred_ids, label_ids, possible_label_ids):
    label_size = possible_label_ids.shape[1]
    new_label_ids = []
    new_pred_ids = []
    for idx, p in enumerate(pred_ids):
        if 0 <= p < label_size and possible_label_ids[idx, p] == 1:
            # match one, then count all
            for idy, item in enumerate(possible_label_ids[idx]):
                if item == 1:
                    new_label_ids.append(idy)
                    new_pred_ids.append(idy)
        else:
            new_pred_ids.append(p)
            new_label_ids.append(label_ids[idx])

    label_ids = np.array(new_label_ids)
    pred_ids = np.array(new_pred_ids)
    acc = accuracy_score(y_true=label_ids, y_pred=pred_ids)
    f1 = f1_score(y_true=label_ids, y_pred=pred_ids, labels=list(range(label_size)), average="macro", zero_division=0)

    return acc, f1

def cal_acc_f1_score_per_label(pred_ids, label_ids, label_list):
    res = classification_report(y_true=label_ids, y_pred=pred_ids, target_names=label_list,
                                labels=list(range(len(label_list))), digits=4, zero_division=0)
    return res


def extract_res_with_keyword(response, keyword):
    pattern = r"<{}>(.*?)</{}>".format(keyword, keyword)
    matches = re.findall(pattern, response, re.DOTALL)
    if matches is None or len(matches) == 0:
        return "None"
    else:
        return matches[-1].strip()


def labels_from_raw_file(task, level, mode):
    data_file = PDTB_SOURCE_DIR + "/{}/{}.json".format(task, mode)
    data_list = read_from_file_json(data_file, json_type=2)
    rel_list, label2idx = get_list_dict_for_task(task, level)
    rel_list = [item.lower() for item in rel_list]
    label_size = len(rel_list)

    label_ids = []
    possible_label_ids = []
    for data in data_list:
        relation_type = data['relation_type']
        if relation_type.lower() != 'implicit':
            continue
        
        relation_class = data['relation_class'].split('##')[0]
        relations = relation_class.split('.')
        if len(relations) < level:
            continue
        rel = relations[level-1].lower()
        if rel not in rel_list:
            continue
        label_ids.append(label2idx[rel])

        all_relation_class = data['relation_class'].split('##')
        tmp_labels = set()
        for label in all_relation_class:
            items = label.split(".")
            if len(items) < level:
                continue
            tmp_labels.add(items[level-1])
        tmp_labels = list(tmp_labels)
        tmp_possible_ids = np.zeros(label_size, dtype=np.int32)
        for label in tmp_labels:
            label = label.lower()
            if label in label2idx:
                idx = label2idx[label]
                tmp_possible_ids[idx] = 1

        possible_label_ids.append(tmp_possible_ids)
    
    return label_ids, np.array(possible_label_ids)



def normalize_pdtb_prediction(text):
    """Text after the last 'Relation:' (lower-cased, without trailing punctuation); '' if there is none."""
    if not text or "relation:" not in text.lower():
        return ""
    return text.lower().split("relation:")[-1].strip().strip(".").strip()


def metric_for_file(data_list, task, level, mode):
    """data_list: one predicted relation string per implicit instance of the split, in the original order
    (use normalize_pdtb_prediction). Predictions that are not a valid label are counted as wrong."""
    label_list, label2idx = get_list_dict_for_task(task, level)
    label2idx = {k.lower(): v for k, v in label2idx.items()}
    all_label_ids, all_possible_label_ids = labels_from_raw_file(task, level, mode)
    assert len(data_list) == len(all_label_ids), \
        f"{len(data_list)} predictions for {len(all_label_ids)} gold instances ({task} level {level}, {mode})"
    invalid_id = len(label_list)  # never a gold label

    pred_ids = []
    n_invalid = 0
    for pred in data_list:
        pred = (pred or "").lower().strip().strip(".").strip()
        if pred in label2idx:
            pred_ids.append(label2idx[pred])
        else:
            pred_ids.append(invalid_id)
            n_invalid += 1
    if n_invalid:
        print(f"{n_invalid} predictions are not a valid label and are counted as wrong.")

    label_ids = np.array(all_label_ids)
    possible_label_ids = np.array(all_possible_label_ids)
    acc, f1 = cal_acc_f1_score_with_ids(pred_ids, label_ids, possible_label_ids)
    acc_v2, f1_v2 = cal_acc_f1_score_with_ids_v2(pred_ids, label_ids, possible_label_ids)
    label_list = [key.lower() for key in label_list]
    res = cal_acc_f1_score_per_label(pred_ids, label_ids, label_list)
    print(res)
    print("E1, Acc: %.4f, F1: %.4f" % (acc, f1))
    print("E2, Acc: %.4f, F1: %.4f" % (acc_v2, f1_v2))
    return acc, f1, acc_v2, f1_v2, res


def resample_loop(current_number, target_number, max_value):
    if current_number == target_number:
        while True:
            new_number = random.randint(0, max_value)
            if new_number != target_number:
                return new_number
    return current_number



"""
=========te eval==========
"""
import os
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import re
import json
from collections import Counter
from collections import OrderedDict
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from preprocess_MATRES import *
#from preprocess_tbd import *

random.seed(42)

import networkx as nx

def _get_inverse_relation(rel_type):
    """Returns the inverse of a temporal relation type."""
    inverses = {
        'before': 'after',
        'after': 'before',
        'is_included_in': 'includes',
        'includes': 'is_included_in',
        'simultaneous': 'simultaneous',
        'equal': 'equal',
    }
    return inverses.get(rel_type, rel_type)

def build_temporal_graph(relations):
    """
    Builds a directed graph from a list of temporal relations.
    Adds inverse relations automatically.
    
    Args:
        relations (list): A list of tuples, e.g., [('e1', 'e2', 'before')].
                          Format: (source_node, target_node, relation_type).

    Returns:
        nx.DiGraph: The constructed temporal graph.
    """
    graph = nx.DiGraph()
    for u, v, rel_type in relations:
        graph.add_edge(u, v, rel=rel_type)
        # Add inverse relation only if it's not a self-loop
        if u != v:
            inv_rel = _get_inverse_relation(rel_type)
            if inv_rel != rel_type:
                graph.add_edge(v, u, rel=inv_rel)
            
    return graph

def get_temporal_closure(graph):
    """
    Computes the temporal closure of the graph.
    Assumes standard temporal transitivity rules.
    
    Args:
        graph (nx.DiGraph): The input temporal graph.
    
    Returns:
        nx.DiGraph: The transitive closure graph.
    """
    return nx.transitive_closure(graph)

def get_reduced_relations(graph, closure):
    """
    Finds the reduced (non-redundant) set of relations.
    A relation (u, v) is redundant if there's a path u -> ... -> v of length > 1 in the original graph.
    
    Args:
        graph (nx.DiGraph): The original graph.
        closure (nx.DiGraph): The transitive closure graph.
    
    Returns:
        list: A list of non-redundant relation tuples.
    """
    reduced_relations = []
    
    for u, v, data in graph.edges(data=True):
        is_redundant = False
        # Check if an indirect path exists from u to v
        if u in closure and v in closure:
            # Check for any path u -> w -> v, where w is an intermediate node
            for w in closure.neighbors(u):
                if w != v and closure.has_edge(w, v):
                    is_redundant = True
                    break
        if not is_redundant:
            reduced_relations.append((u, v, data['rel']))

    return reduced_relations

def calculate_metrics_tg(preds, labels):
    """
    Calculates Precision, Recall, and F1 based on the paper's graph-based metric.
    
    Args:
        preds (list): Predicted temporal relations.
        labels (list): True (gold) temporal relations.
    
    Returns:
        tuple: Precision, Recall, and F1 scores.
    """
    sys_graph = build_temporal_graph(preds)
    ref_graph = build_temporal_graph(labels)
    
    sys_closure = get_temporal_closure(sys_graph)
    ref_closure = get_temporal_closure(ref_graph)
    
    sys_reduced = set(get_reduced_relations(sys_graph, sys_closure))
    ref_reduced = set(get_reduced_relations(ref_graph, ref_closure))

    # Calculate Precision
    tp_sys = 0
    for u, v, rel in sys_reduced:
        if ref_closure.has_edge(u, v):
            tp_sys += 1
    
    precision = tp_sys / len(sys_reduced) if len(sys_reduced) > 0 else 0

    # Calculate Recall
    tp_ref = 0
    for u, v, rel in ref_reduced:
        if sys_closure.has_edge(u, v):
            tp_ref += 1

    recall = tp_ref / len(ref_reduced) if len(ref_reduced) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def calculate_metrics_with_vague(preds, labels, target_labels):
    """
    Calculates Precision, Recall, and F1-Score by handling a special 'vague' label.

    The 'vague' label is excluded from the calculation of TP, FP, and FN.
    This logic breaks the typical multi-class micro-averaging assumption
    (sum(FP) == sum(FN)), leading to distinct PRF scores.

    Args:
        preds (list): A list of predicted labels.
        labels (list): A list of true labels.
        target_labels (set): A set of labels to be considered in the calculation.
    
    Returns:
        tuple: A tuple containing (precision, recall, f1_score).
    """
    tp = 0
    fp = 0
    fn = 0
    
    for i in range(len(preds)):
        pred = preds[i]
        true_label = labels[i]
        
        is_pred_target = pred in target_labels
        is_true_target = true_label in target_labels
        
        if is_pred_target and is_true_target:
            if pred == true_label:
                # Correct prediction of a target label
                tp += 1
            else:
                # Mismatched prediction of a target label
                fp += 1
                fn += 1
        elif is_pred_target and not is_true_target:
            # Predicted a target label, but the true label is 'vague'
            # This counts as a false positive
            fp += 1
        elif not is_pred_target and is_true_target:
            # Predicted 'vague', but the true label is a target label
            # This counts as a false negative
            fn += 1

    # Handle division by zero for robustness
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
    

"""
def calculate_metrics_classification(preds, labels, mode):
    if mode == "matres":
        label_to_idx = {'before': 0, 'after': 1, 'equal': 2, 'vague': 3}
        top_k = 3
    else:
        label_to_idx = {
            'before': 0, 'after': 1, 'is_included': 2, 
            'includes': 3, 'simultaneous': 4, "vague": 5
        }
        top_k = 5

    num_classes = len(label_to_idx)
    vague_idx = num_classes - 1
    
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    for p_rel, g_rel in zip(preds, labels):
        if p_rel in label_to_idx and g_rel in label_to_idx:
            p_idx = label_to_idx[p_rel]
            g_idx = label_to_idx[g_rel]
            confusion[g_idx][p_idx] += 1
        else:
            confusion[vague_idx][vague_idx] += 1

    
    correct_core = np.sum([confusion[i][i] for i in range(top_k)])
    
    precision_denom = np.sum(confusion) - np.sum(confusion, axis=0)[vague_idx]
    test_prec = correct_core / precision_denom if precision_denom > 0 else 0
    
    recall_denom = np.sum(confusion) - np.sum(confusion[vague_idx, :])
    test_rec = correct_core / recall_denom if recall_denom > 0 else 0
    
    if (test_prec + test_rec) > 0:
        test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec)
    else:
        test_f1 = 0
        
    return test_prec, test_rec, test_f1, confusion
"""

def calculate_metrics_classification(preds, labels, mode="matres"):
    """
    Compute metrics following the logic of eval_temporal.py.
    
    Args:
        preds (list): predicted label strings, e.g. ['before', 'none', 'equal']
        labels (list): gold label strings, e.g. ['before', 'vague', 'equal']
        mode (str): evaluation mode; currently mainly 'matres'
    """
    
    if mode == "matres":
        rel2id = {"before": 0, "after": 1, "equal": 2, "vague": 3}
        eval_labels = [0, 1, 2]
        target_names = ["BEFORE", "AFTER", "EQUAL"]
    else:
        rel2id = {"before": 0, "after": 1, "includes": 2, "is_included": 3, "simultaneous": 4, "vague": 5}
        eval_labels = [0, 1, 2, 3, 4]
        target_names = ["BEFORE", "AFTER", "INCLUDES", "IS_INCLUDED", "SIMULTANEOUS"]

    # VAGUE is excluded from the micro-averaged scores; a prediction that is not a valid label is treated
    # like VAGUE (it is not counted as a positive prediction, and the gold relation counts as missed).
    invalid_id = len(rel2id)
    norm = lambda x: (x or "").lower().strip().strip(".").strip()
    y_pred = [rel2id.get(norm(p), invalid_id) for p in preds]
    unknown_gold = sorted({norm(l) for l in labels if norm(l) not in rel2id})
    assert not unknown_gold, f"unknown gold labels for {mode}: {unknown_gold}"
    y_true = [rel2id[norm(l)] for l in labels]

    report = classification_report(
        y_true, 
        y_pred, 
        labels=eval_labels, 
        target_names=target_names, 
        output_dict=True,
        zero_division=0
    )

    results = {}
    for name in target_names:
        results[name] = {
            "precision": round(report[name]["precision"] * 100, 2),
            "recall": round(report[name]["recall"] * 100, 2),
            "f1-score": round(report[name]["f1-score"] * 100, 2),
            "support": int(report[name]["support"])
        }
    
    results["micro_avg"] = {
        "precision": round(report["micro avg"]["precision"] * 100, 2),
        "recall": round(report["micro avg"]["recall"] * 100, 2),
        "f1-score": round(report["micro avg"]["f1-score"] * 100, 2),
        "support": int(report["micro avg"]["support"])
    }

    return results


# Kept for backward compatibility: zero-shot predictions use the same metric.
calculate_metrics_classification_zs = calculate_metrics_classification

if __name__ == "__main__":
    #file_path = "LLaMA-Factory/saves/Llama-3.2-3B-Instruct/lora/eval_2025-03-16-20-38-09/generated_predictions.jsonl"
    #file_path = "saves/Llama-3.3-70B-Instruct/lora/eval_2025-03-17-22-47-35/generated_predictions.jsonl"

    task = "pdtb3"
    level = 2
    eval_mode = "test"
    #metric_for_file(file_path, task, level, mode, "single")

    #metric_for_file_v2(file_path, task, level, mode, "single")

    """
    ds_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/gen_cyclic_stage2_conn_rel_matres4pdtb2_l1_test_1/generated_predictions.jsonl"
    te_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/stage2_conn_rel_pdtb2l1_tbd_test_te_1/generated_predictions.jsonl"
    """

    ds_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res_stage2/gen_cyclic_stage2_conn_rel_matres4pdtb3_l2_test_2/generated_predictions.jsonl"

    te_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/rel_matres_test_zs_2/generated_predictions.jsonl"
    #te_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/rel_matres_pdtb2_l1_test_te_2/generated_predictions.jsonl"

    te_data_list = read_from_file_json(te_file, json_type=2)
    ds_data_list = read_from_file_json(ds_file, json_type=2)

    te_pred, ds_pred = [], []
    te_labels = []

    # one prediction per instance, so that predictions stay aligned with the gold labels
    for data in ds_data_list:
        ds_pred.append(normalize_pdtb_prediction(data['predict']))

    for data in te_data_list:
        te_pred.append(normalize_pdtb_prediction(data['predict']))
        te_labels.append(normalize_pdtb_prediction(data['label']))

    if "zs" not in te_file:
        if "matres" in te_file:
            results = calculate_metrics_classification(te_pred, te_labels, "matres")
        else:
            results = calculate_metrics_classification(te_pred, te_labels, "tbd")
    else:
        if "matres" in te_file:
            results = calculate_metrics_classification_zs(te_pred, te_labels, "matres")
        else:
            results = calculate_metrics_classification_zs(te_pred, te_labels, "tbd")


    acc, f1, acc_v2, f1_v2, res = metric_for_file(ds_pred, task, level, eval_mode)

    print()
    #print(round(precision_all * 100, 2), round(recall_all * 100, 2), round(f1_all * 100, 2))
    print(te_pred[:5])
    print(te_labels[:5])
    #print(precision_denom)
    print(results["micro_avg"])
    #print(results)
