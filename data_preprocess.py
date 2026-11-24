import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random
import textwrap
from transformers import get_linear_schedule_with_warmup
import re
import json
from tqdm import trange, tqdm

from common import split_sentence_by_entities
from paths import LLAMA_FACTORY_DIR

L1_LABELS = "Temporal, Contingency, Comparison, or Expansion"
PDTB2_L2_LABELS = "Concession, Contrast, Cause, Pragmatic cause, Conjunction, Instantiation, Alternative, List, Restatement, Asynchronous or Synchrony"
PDTB3_L2_LABELS = "Concession, Contrast, Cause, Cause+Belief, Condition, Purpose, Conjunction, Equivalence, Instantiation, Level-of-detail, Manner, Substitution, Asynchronous or Synchronous"

PDTB2L2_RELS = ["Concession", "Contrast", "Cause", "Pragmatic cause", "Conjunction", "Instantiation", "Alternative", "List", "Restatement", "Asynchronous", "Synchrony"]
PDTB3L2_RELS = ["Concession", "Contrast", "Cause", "Cause+Belief", "Condition", "Purpose", "Conjunction", "Equivalence", "Instantiation", "Level-of-detail", "Manner", "Substitution", "Asynchronous", "Synchronous"]


def preprocess_pdtb_init(pdtb_data_list, pdtb_type, output_dir, mode):
    pdtb_conn_rel_data, pdtb_rel_data = [], []
    if "l1" in pdtb_type:
        label_set = L1_LABELS
    elif pdtb_type == "pdtb2_l2":
        label_set = PDTB2_L2_LABELS
    else:
        label_set = PDTB3_L2_LABELS

    for i in range(len(pdtb_data_list)):
        data = pdtb_data_list[i]
        arg1 = data["arg1"]
        arg2 = data["arg2"]
        conn = data['conn']
        conn_list = conn.split("##")
        rel = data["rel"]
        conn_rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2. First, identify a potential connective that fits the gap. Then, determine the discourse relation label. 
        Output your reasoning about the potential connectiveand the discourse relation. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Connective: [Your predicted connective], Relation: [Your predicted relation]""")

        rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2 and determine the discourse relation label. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Relation: [Your predicted relation]""")

        input_text = textwrap.dedent(f"""Input text:
        Arg1: {arg1}
        Arg2: {arg2}""")

        conn_rel_text = f"Connective: {conn_list[0]}, Relation: {rel}"

        rel_text = f"Relation: {rel}"

        conn_rel_input = {
            'instruction':conn_rel_instruction,
            'input':input_text,
            'output':conn_rel_text,
        }

        rel_input = {
            'instruction':rel_instruction,
            'input':input_text,
            'output':rel_text,
        }

        pdtb_conn_rel_data.append(conn_rel_input)
        pdtb_rel_data.append(rel_input)

    conn_rel_file = output_dir + "conn_rel_{}_{}.json".format(pdtb_type, mode)
    rel_file = output_dir + "rel_{}_{}.json".format(pdtb_type, mode)

    with open(conn_rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(pdtb_conn_rel_data, json_file, ensure_ascii=False, indent=4)

    with open(rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(pdtb_rel_data, json_file, ensure_ascii=False, indent=4)


def preprocess_te_init(te_data_list, te_task, output_dir, mode):
    conn_rel_data, conn_data, rel_data = [], [], []
    if te_task == "matres":
        label_inst = "The Result MUST be EXACTLY one of: before, after, equal and vague."
        te_info = "before, after, equal and vague"
    else:
        label_inst = "The Result MUST be EXACTLY one of: before, after, is_included, includes, simultaneous and vague."
        te_info = "before, after, is_included, includes, simultaneous and vague"

    for i in range(len(te_data_list)):
            data = te_data_list[i]

            tagged_e1 = data["tagged_e1"]
            tagged_e2 = data["tagged_e2"]
            e1_sent_id, e2_sent_id = data["event1_sent_id"], data["event2_sent_id"]
            e1_sent, e2_sent = data["event1_sentence"], data["event2_sentence"]

            relation = data["relation"].lower()
            #context_sents = data["context_sents"]
            conn = None

            if e1_sent_id != e2_sent_id:
                arg1 = e1_sent
                arg2 = e2_sent
            else:
                split_res = split_sentence_by_entities(e1_sent)
                [arg1, arg2] = split_res["split_tagged"]
                conn = split_res["connective"]

            conn_rel_instruction = textwrap.dedent(f"""
            Your task is to analyze the text logic by identifying a potential connective that fits the gap, and then determine the temporal relation between Event {tagged_e1} and Event {tagged_e2}. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
            The output format shoud be:
            Connective: [Your predicted connective], Relation: [Your predicted relation].""")

            rel_instruction = textwrap.dedent(f"""
            Your task is to analyze the text of Arg1 and Arg2 and then determine the temporal relation between Event {tagged_e1} and Event {tagged_e2}. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
            The output format shoud be:
            Relation: [Your predicted relation]""")

            conn_instruction = textwrap.dedent(f"""
            Your task is to analyze the text logic by identifying a potential connective that fits the gap (if there's no connective between Arg1 and Arg2).
            The output format shoud be:
            Connective: [Your predicted connective]""")

            input_text = textwrap.dedent(f"""Input text:
            Arg1: {arg1}
            Arg2: {arg2}""")

            conn_label_text = f"Connective: {conn}, Relation: {relation}"

            rel_text = f"Relation: {relation}"

            conn_text = f"Connective: {conn}"

            conn_rel_input = {
            'instruction':conn_rel_instruction,
            'input':input_text,
            'output':conn_label_text,
        }
            conn_input = {
            'instruction':conn_instruction,
            'input':input_text,
            'output':conn_text,
        }
            rel_input = {
            'instruction':rel_instruction,
            'input':input_text,
            'output':rel_text,
        }

            conn_rel_data.append(conn_rel_input)
            rel_data.append(rel_input)
            conn_data.append(conn_input)

    conn_rel_file = output_dir + "conn_rel_{}_{}.json".format(te_task, mode)
    rel_file = output_dir + "rel_{}_{}.json".format(te_task, mode)
    conn_file = output_dir + "conn_{}_{}.json".format(te_task, mode)

    with open(conn_rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(conn_rel_data, json_file, ensure_ascii=False, indent=4)

    with open(rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(rel_data, json_file, ensure_ascii=False, indent=4)
    
    with open(conn_file, 'w', encoding='utf-8') as json_file:
        json.dump(conn_data, json_file, ensure_ascii=False, indent=4)


def preprocess_te_exp_init(te_data_list, te_task, output_dir, mode):
    conn_rel_data, conn_data, rel_data = [], [], []
    if te_task == "matres":
        label_inst = "The Result MUST be EXACTLY one of: before, after, equal and vague."
        te_info = "before, after, equal and vague"
    else:
        label_inst = "The Result MUST be EXACTLY one of: before, after, is_included, includes, simultaneous and vague."
        te_info = "before, after, is_included, includes, simultaneous and vague"

    for i in range(len(te_data_list)):
            data = te_data_list[i]

            tagged_e1 = data["tagged_e1"]
            tagged_e2 = data["tagged_e2"]
            e1_sent_id, e2_sent_id = data["event1_sent_id"], data["event2_sent_id"]
            e1_sent, e2_sent = data["event1_sentence"], data["event2_sentence"]
            context_sents = data['context_sents']

            relation = data["relation"].lower()

            #context_sents = data["context_sents"]
            conn = None

            if e1_sent_id != e2_sent_id:
                arg1 = e1_sent
                arg2 = e2_sent
            else:
                split_res = split_sentence_by_entities(e1_sent)
                [arg1, arg2] = split_res["split_tagged"]
                conn = split_res["connective"]

            conn_rel_instruction = textwrap.dedent(f"""
            Based on the context, your task is to analyze the text logic by identifying a potential connective that fits the gap, and then determine the temporal relation between Event {tagged_e1} and Event {tagged_e2}. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
            The output format shoud be:
            Connective: [Your predicted connective], Relation: [Your predicted relation].""")

            rel_instruction = textwrap.dedent(f"""
            Based on the context, your task is to analyze the text of Arg1 and Arg2 and then determine the temporal relation between Event {tagged_e1} and Event {tagged_e2}. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
            The output format shoud be:
            Relation: [Your predicted relation]""")

            conn_instruction = textwrap.dedent(f"""
            Based on the context, your task is to analyze the text logic by identifying a potential connective that fits the gap (if there's no connective between Arg1 and Arg2).
            The output format shoud be:
            Connective: [Your predicted connective]""")

            input_text = textwrap.dedent(f"""Input text:
            Context: {context_sents}
            Arg1: {arg1}
            Arg2: {arg2}""")

            conn_label_text = f"Connective: {conn}, Relation: {relation}"

            rel_text = f"Relation: {relation}"

            conn_text = f"Connective: {conn}"

            conn_rel_input = {
            'instruction':conn_rel_instruction,
            'input':input_text,
            'output':conn_label_text,
        }
            conn_input = {
            'instruction':conn_instruction,
            'input':input_text,
            'output':conn_text,
        }
            rel_input = {
            'instruction':rel_instruction,
            'input':input_text,
            'output':rel_text,
        }

            conn_rel_data.append(conn_rel_input)
            rel_data.append(rel_input)
            conn_data.append(conn_input)

    conn_rel_file = output_dir + "context_conn_rel_{}_{}.json".format(te_task, mode)
    rel_file = output_dir + "context_rel_{}_{}.json".format(te_task, mode)
    conn_file = output_dir + "context_conn_{}_{}.json".format(te_task, mode)

    with open(conn_rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(conn_rel_data, json_file, ensure_ascii=False, indent=4)

    with open(rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(rel_data, json_file, ensure_ascii=False, indent=4)
    
    with open(conn_file, 'w', encoding='utf-8') as json_file:
        json.dump(conn_data, json_file, ensure_ascii=False, indent=4)


def init_cyclic_stage_1(pdtb_data_list, pdtb_type, te_data_list, te_task, output_dir, mode):
    te4pdtb_data, detailed_te4pdtb_data, pdtb4te_conn_rel_data, pdtb4te_rel_data = [], [], [], []

    if "l1" in pdtb_type:
        label_set = L1_LABELS
    elif pdtb_type == "pdtb2_l2":
        label_set = PDTB2_L2_LABELS
    else:
        label_set = PDTB3_L2_LABELS

    if te_task == "matres":
        label_inst = "The Result MUST be EXACTLY one of: before, after, equal and vague."
        te_info = "before, after, equal and vague"
    else:
        label_inst = "The Result MUST be EXACTLY one of: before, after, is_included, includes, simultaneous and vague."
        te_info = "before, after, is_included, includes, simultaneous and vague"

    for i in range(len(pdtb_data_list)):
        data = pdtb_data_list[i]
        arg1 = data["arg1"]
        arg2 = data["arg2"]

        rel = data["rel"]

        te_instruction = textwrap.dedent(f"""
        Your task is to analyze the text of Arg1 and Arg2 and identify the core events (mainly verb, time expression, or noun phrase). Then determine the temporal relation between core events. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
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

        """)

        detailed_instruction = textwrap.dedent(f"""
        You are a temporal logic analyst, specializing in the subtle connections between discourse relations and temporal relations. Your task is to identify the core events within the two provided arguments and determine the precise temporal relation between them.

        ## Your Workflow:
        1.  **Identify Core Events:** In each argument, locate the main events (mainly verb, time expression or noun phrase) that represents the core event. Consider the most significant action or state. Label the core event in Arg1 as **Event 1** and the core event in Arg2 as **Event 2**.
        2.  **Analyze Temporal Relation:** Based on linguistic cues (e.g., temporal expressions, verb tense) and logical inference, determine the temporal relationship between Event 1 and Event 2.
        3.  **Strict Output Format:** You must present your answer in a concise and structured format, focusing solely on the identified events and their temporal relation, possible labels: {te_info}

        ## Output Format
        (Event 1, Event 2, RELATION)

        **Example:**
        Input:
        Arg1: He was brought to the company in September 1988 after 15 years at First Interstate Bancorp.
        Arg2: The bank had been suffering in late 1987 from a slew of bad real estate loans.

        Output:
        (brought, suffering, after); (September 1988, late 1987, after)
        """)


        input_text = textwrap.dedent(f"""Input text:
        Arg1: {arg1}
        Arg2: {arg2}""")

        te4pdtb_input = {
            'instruction':te_instruction,
            'input':input_text,
            'output':rel,
        }

        de_te4pdtb_input = {
            'instruction':detailed_instruction,
            'input':input_text,
            'output':rel,
        }

        te4pdtb_data.append(te4pdtb_input)
        detailed_te4pdtb_data.append(de_te4pdtb_input)


    for i in range(len(te_data_list)):
        data = te_data_list[i]

        tagged_e1 = data["tagged_e1"]
        tagged_e2 = data["tagged_e2"]
        e1_sent_id, e2_sent_id = data["event1_sent_id"], data["event2_sent_id"]
        e1_sent, e2_sent = data["event1_sentence"], data["event2_sentence"]

        relation = data["relation"].lower().strip()
        rel = relation

        conn = None

        if e1_sent_id != e2_sent_id:
            arg1 = e1_sent
            arg2 = e2_sent
        else:
            split_res = split_sentence_by_entities(e1_sent)
            [arg1, arg2] = split_res["split_tagged"]
            conn = split_res["connective"]

        pdtb_conn_rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2. First, identify a potential connective that fits the gap. Then, determine the discourse relation label. 
        Output your reasoning about the potential connectiveand the discourse relation. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Connective: [Your predicted connective], Relation: [Your predicted relation]""")

        pdtb_rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2 and determine the discourse relation label. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Relation: [Your predicted relation]""")

        input_text = textwrap.dedent(f"""Input text:
            Arg1: {arg1}
            Arg2: {arg2}""")
        
        pdtb4te_conn_rel_input = {
            'instruction':pdtb_conn_rel_instruction,
            'input':input_text,
            'output':rel,
        }

        pdtb4te_rel_input = {
            'instruction':pdtb_rel_instruction,
            'input':input_text,
            'output':rel,
        }

        pdtb4te_conn_rel_data.append(pdtb4te_conn_rel_input)
        pdtb4te_rel_data.append(pdtb4te_rel_input)

    te4pdtb_file = output_dir + "cyclic_stage1_{}_{}_{}.json".format(pdtb_type, te_task, mode)
    de_te4pdtb_file = output_dir + "cyclic_stage1_detailed_{}_{}_{}.json".format(pdtb_type, te_task, mode)

    pdtb4te_conn_rel_file = output_dir + "cyclic_stage1_conn_rel_{}_{}_{}.json".format(te_task, pdtb_type, mode)
    pdtb4te_rel_file = output_dir + "cyclic_stage1_rel_{}_{}_{}.json".format(te_task, pdtb_type, mode)

    with open(de_te4pdtb_file, 'w', encoding='utf-8') as json_file:
        json.dump(detailed_te4pdtb_data, json_file, ensure_ascii=False, indent=4)
    
    with open(te4pdtb_file, 'w', encoding='utf-8') as json_file:
        json.dump(te4pdtb_data, json_file, ensure_ascii=False, indent=4)
    
    with open(pdtb4te_conn_rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(pdtb4te_conn_rel_data, json_file, ensure_ascii=False, indent=4)
    
    with open(pdtb4te_rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(pdtb4te_rel_data, json_file, ensure_ascii=False, indent=4)

    print("File saved in: " + de_te4pdtb_file)
    print("Data for cyclic stage1 is saved!")




def init_cyclic_stage_2_pdtb4te(stage1_rel_output_file, stage1_conn_rel_output_file, te_data_list, output_dir, te_task, pdtb_type, mode, model_seq):
    #(pdtb_data_list, pdtb_type, te_data_list, te_task, te4pdtb_list, pdtb4te_list, output_dir, mode):
    pdtb4te_rel_data, pdtb4te_conn_rel_data = [], []

    pred_rel_pdtb_data, pred_conn_rel_pdtb_data = [], []
    with open(stage1_rel_output_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())  
            pred_rel_pdtb_data.append(data["predict"])
    
    with open(stage1_conn_rel_output_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())  
            pred_conn_rel_pdtb_data.append(data["predict"])
    
    assert len(te_data_list) == len(pred_rel_pdtb_data)
    assert len(te_data_list) == len(pred_conn_rel_pdtb_data)

    if te_task == "matres":
        label_inst = "The Result MUST be EXACTLY one of: before, after, equal and vague."
        te_info = "before, after, equal and vague"
    else:
        label_inst = "The Result MUST be EXACTLY one of: before, after, is_included, includes, simultaneous and vague."
        te_info = "before, after, is_included, includes, simultaneous and vague"

    for i in range(len(te_data_list)):
            data = te_data_list[i]
            pdtb_pred_rel_info = pred_rel_pdtb_data[i].replace("<think>\n\n</think>\n\n", "")
            pdtb_pred_conn_rel_info = pred_conn_rel_pdtb_data[i].replace("<think>\n\n</think>\n\n", "")

            tagged_e1 = data["tagged_e1"]
            tagged_e2 = data["tagged_e2"]
            e1_sent_id, e2_sent_id = data["event1_sent_id"], data["event2_sent_id"]
            e1_sent, e2_sent = data["event1_sentence"], data["event2_sentence"]
            context_sents = data['context_sents']

            relation = data["relation"].lower()

            #context_sents = data["context_sents"]
            conn = None

            relation = data["relation"].lower().strip()
            rel = relation

            if e1_sent_id != e2_sent_id:
                arg1 = e1_sent
                arg2 = e2_sent
            else:
                split_res = split_sentence_by_entities(e1_sent)
                [arg1, arg2] = split_res["split_tagged"]
                conn = split_res["connective"]

            rel_instruction = textwrap.dedent(f"""
            Based on the context, your task is to analyze the text of Arg1 and Arg2 and then determine the temporal relation between Event {tagged_e1} and Event {tagged_e2}. You will be provided a predicted discourse relation between Arg1 and Arg2 to help you predict the temporal relation between Event {tagged_e1} and Event {tagged_e2}. You can only refer to it, but not fully trust this prediction. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
            The output format shoud be:
            Relation: [Your predicted relation]""")

            conn_rel_instruction = textwrap.dedent(f"""
            Based on the context, your task is to analyze the text of Arg1 and Arg2 and then determine the temporal relation between Event {tagged_e1} and Event {tagged_e2}. You will be provided a predicted connective and a discourse relation between Arg1 and Arg2 to help you predict the temporal relation between Event {tagged_e1} and Event {tagged_e2}. You can only refer to it, but not fully trust this prediction. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
            The output format shoud be:
            Relation: [Your predicted relation]""")

            input_text = textwrap.dedent(f"""Input text:
            Context: {context_sents}
            Arg1: {arg1}
            Arg2: {arg2}
            Predicted Discourse {pdtb_pred_rel_info}  
            """)

            conn_input_text = textwrap.dedent(f"""Input text:
            Context: {context_sents}
            Arg1: {arg1}
            Arg2: {arg2}
            Predicted Connective and Discourse Relation: {pdtb_pred_conn_rel_info}
            """)


            #rel_text = f"Relation: {relation}"
            rel_text = f"Relation: {rel}"

            rel_input = {
            'instruction':rel_instruction,
            'input':input_text,
            'output':rel_text,
            }
            conn_rel_input = {
            'instruction':conn_rel_instruction,
            'input':conn_input_text,
            'output':rel_text,
            }

            pdtb4te_rel_data.append(rel_input)
            pdtb4te_conn_rel_data.append(conn_rel_input)

    conn_rel_file = output_dir + "cyclic_stage2_conn_rel_{}4{}_{}_{}.json".format(pdtb_type, te_task, mode, model_seq)
    rel_file = output_dir + "cyclic_stage2_rel_{}4{}_{}_{}.json".format(pdtb_type, te_task, mode, model_seq)

    """with open(conn_rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(pdtb4te_rel_data, json_file, ensure_ascii=False, indent=4)"""

    with open(rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(pdtb4te_conn_rel_data, json_file, ensure_ascii=False, indent=4)
    
    print("File saved in: " + rel_file)
    print("File saved in: " + conn_rel_file)


def init_cyclic_stage_2_te4pdtb(stage1_rel_output_file, pdtb_data_list, output_dir, te_task, pdtb_type, mode, model_seq, hq):
    #(pdtb_data_list, pdtb_type, te_data_list, te_task, te4pdtb_list, pdtb4te_list, output_dir, mode):
    te4pdtb_rel_data, te4pdtb_conn_rel_data = [], []
    pred_rel_te_data = []

    if not hq:
        pred_rel_te_data = extract_te_predictions(stage1_rel_output_file)
    else:
        with open(stage1_rel_output_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line.strip())  
                pred_rel_te_data.append(data["predict"])
    
    print(stage1_rel_output_file)
    print(len(pdtb_data_list))
    print(len(pred_rel_te_data))
    assert len(pdtb_data_list) == len(pred_rel_te_data)

    if "l1" in pdtb_type:
        label_set = L1_LABELS
    elif pdtb_type == "pdtb2_l2":
        label_set = PDTB2_L2_LABELS
    else:
        label_set = PDTB3_L2_LABELS
        
    for i in range(len(pdtb_data_list)):
        pred_te = pred_rel_te_data[i]

        data = pdtb_data_list[i]
        arg1 = data["arg1"]
        arg2 = data["arg2"]
        conn = data['conn']
        conn_list = conn.split("##")
        rel = data["rel"]

        conn_rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2. First, identify a potential connective that fits the gap. Then, determine the discourse relation label. You will be provided a predicted temporal relation between events in Arg1 and Arg2 to help you predict the discourse relation. You can only refer to it, but not fully trust this prediction.
        Output your reasoning about the potential connectiveand the discourse relation. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Connective: [Your predicted connective], Relation: [Your predicted relation]""")

        rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2 and determine the discourse relation label. You will be provided a predicted temporal relation between events in Arg1 and Arg2 to help you predict the discourse relation. You can only refer to it, but not fully trust this prediction. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Relation: [Your predicted relation]""")


        input_text = textwrap.dedent(f"""Input text:
        Arg1: {arg1}
        Arg2: {arg2}

        Predicted temporal relation: {pred_te}
        """)

        conn_rel_text = f"Connective: {conn_list[0]}, Relation: {rel}"

        rel_text = f"Relation: {rel}"

        conn_rel_input = {
            'instruction':conn_rel_instruction,
            'input':input_text,
            'output':conn_rel_text,
        }

        rel_input = {
            'instruction':rel_instruction,
            'input':input_text,
            'output':rel_text,
        }

        te4pdtb_conn_rel_data.append(conn_rel_input)
        te4pdtb_rel_data.append(rel_input)


    conn_rel_file = output_dir + "cyclic_stage2_conn_rel_{}4{}_{}_{}.json".format(te_task, pdtb_type, mode, model_seq)
    rel_file = output_dir + "cyclic_stage2_rel_{}4{}_{}_{}.json".format(te_task, pdtb_type, mode, model_seq)

    with open(conn_rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(te4pdtb_conn_rel_data, json_file, ensure_ascii=False, indent=4)

    with open(rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(te4pdtb_rel_data, json_file, ensure_ascii=False, indent=4)

    print("File saved in: " + rel_file)
    print("File saved in: " + conn_rel_file)




def init_data_preprocess():
    pdtb_type = ['pdtb2_l1', 'pdtb2_l2', 'pdtb3_l1', 'pdtb3_l2']
    te_task = ['tbd', 'matres']
    output_dir = LLAMA_FACTORY_DIR + "/cyclic_data/"
    modes = ["train", "test"]

    for pt in pdtb_type:
        for mode in modes:
            file_name = "./data/pdtb_data/implicit_{}_{}.json".format(pt, mode)
            with open(file_name, 'r', encoding='utf-8') as f:
                pdtb_data_list = json.load(f)
            preprocess_pdtb_init(pdtb_data_list, pt, output_dir, mode)

    for te in te_task:
        for mode in modes:
            file_name = "./data/te_data/{}_{}_data.json".format(te, mode)
            with open(file_name, 'r', encoding='utf-8') as f:
                te_data_list = json.load(f)
            preprocess_te_init(te_data_list, te, output_dir, mode)

    for te in te_task:
        for mode in modes:
            file_name = "./data/te_data/{}_{}_data.json".format(te, mode)
            with open(file_name, 'r', encoding='utf-8') as f:
                te_data_list = json.load(f)
            preprocess_te_exp_init(te_data_list, te, output_dir, mode)

    for pt in pdtb_type:
        for te in te_task:
            for mode in modes:
                file_name = "./data/pdtb_data/implicit_{}_{}.json".format(pt, mode)
                with open(file_name, 'r', encoding='utf-8') as f:
                    pdtb_data_list = json.load(f)

                file_name = "./data/te_data/{}_{}_data.json".format(te, mode)
                with open(file_name, 'r', encoding='utf-8') as f:
                    te_data_list = json.load(f)

                init_cyclic_stage_1(pdtb_data_list, pt, te_data_list, te, output_dir, mode)


def init_stage2_data_preprocess():
    te_tasks = ["matres", "tbd"]
    model_seq = [1, 2]
    model_seq2 = [1, 2, 3, 4]
    pdtb_types = ['pdtb2_l1', 'pdtb2_l2', 'pdtb3_l1', 'pdtb3_l2']
    modes = ["train", "test"]
    output_dir = LLAMA_FACTORY_DIR + "/cyclic_stage2_data/"
    for te in te_tasks:
        for pt in pdtb_types:
            for mode in modes:
                for mos in model_seq:
                    pt_f = pt[:-1] + "2"
                    stage1_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/cyclic_stage1_rel_{}_{}_{}_{}/generated_predictions.jsonl".format(te, pt_f, mode, mos)
                    stage1_conn_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/cyclic_stage1_conn_rel_{}_{}_{}_{}/generated_predictions.jsonl".format(te, pt_f, mode, mos+2)

                    file_name = "./data/te_data/{}_{}_data.json".format(te, mode)
                    with open(file_name, 'r', encoding='utf-8') as f:
                        te_data_list = json.load(f)

                    init_cyclic_stage_2_pdtb4te(stage1_rel_output_file, stage1_conn_rel_output_file, te_data_list, output_dir, te, pt, mode, mos)

                for mos2 in model_seq2:
                    pt_f = pt
                    stage1_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/cyclic_stage1_de_{}_sim_{}_{}_{}/generated_predictions.jsonl".format(pt_f, te, mode, mos2)
                    file_name = "./data/pdtb_data/implicit_{}_{}.json".format(pt, mode)

                    with open(file_name, 'r', encoding='utf-8') as f:
                        pdtb_data_list = json.load(f)

                    init_cyclic_stage_2_te4pdtb(stage1_rel_output_file, pdtb_data_list, output_dir, te, pt, mode, mos2, False)

                hq_file = LLAMA_FACTORY_DIR + "/saves/Llama-3.3-70B-Instruct/te4pdtb_res/{}_{}_sup_{}/generated_predictions.jsonl".format(pt_f, te, mode)

                init_cyclic_stage_2_te4pdtb(hq_file, pdtb_data_list, output_dir+"hq", te, pt, mode, mos2, True)


def init_stage2_data_gen_preprocess():
    te_tasks = ["matres", "tbd"]
    model_seq = [1, 2]
    pdtb_types = ['pdtb2_l1', 'pdtb2_l2', 'pdtb3_l1', 'pdtb3_l2']
    modes = ["train", "test"]
    output_dir = LLAMA_FACTORY_DIR + "/cyclic_stage2_gen_data/"
    for te in te_tasks:
        for pt in pdtb_types:
            for mode in modes:
                for mos in model_seq:

                    #stage1_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/cyclic_stage1_rel_{}_{}_{}_{}/generated_predictions.jsonl".format(te, pt_f, mode, mos)
                    stage1_conn_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res_stage2/dym_conn_rel_{}_{}_{}_{}/generated_predictions.jsonl".format(te, pt, mode, mos)

                    file_name = "./data/te_data/{}_{}_data.json".format(te, mode)
                    with open(file_name, 'r', encoding='utf-8') as f:
                        te_data_list = json.load(f)

                    init_cyclic_stage_2_pdtb4te(stage1_conn_rel_output_file, stage1_conn_rel_output_file, te_data_list, output_dir, te, pt, mode, mos)

                    stage1_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res_stage2/dym_conn_rel_{}_{}_{}_{}/generated_predictions.jsonl".format(pt, te, mode, mos)
                    file_name = "./data/pdtb_data/implicit_{}_{}.json".format(pt, mode)

                    with open(file_name, 'r', encoding='utf-8') as f:
                        pdtb_data_list = json.load(f)

                    init_cyclic_stage_2_te4pdtb(stage1_rel_output_file, pdtb_data_list, output_dir, te, pt, mode, mos, False)

                #hq_file = LLAMA_FACTORY_DIR + "/saves/Llama-3.3-70B-Instruct/te4pdtb_res/{}_{}_sup_{}/generated_predictions.jsonl".format(pt_f, te, mode)

                #init_cyclic_stage_2_te4pdtb(hq_file, pdtb_data_list, output_dir+"hq", te, pt, mode, mos2, True)


def extract_te_predictions(file_path):
    """
    Read a JSONL prediction file and extract '(arg1, arg2, rel)' tuples from the 'predict' field.
    Returns a list of the extracted tuple strings.
    """
    extracted_data = []
    
    pattern = r"\(([^)]+)\)"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                match_srt = ""
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    predict_text = data.get("predict", "")
                    
                    matches = re.findall(pattern, predict_text.replace("<think>\n\n</think>\n\n", ""))
                    
                    for match in matches:
                        match_srt = match_srt + " " + f"({match})"
                    
                    extracted_data.append(match_srt)
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                    
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}")

    return extracted_data

def random_replace_predicted_rel(predicted_rel, rel_type):
    """
    Randomly replace a predicted relation label with a different label
    from a predefined label set.
    """
    tbd_possible_rels = ['before', 'after', 'is_included', 'includes', 'simultaneous', 'vague']
    matres_possible_rels = ['before', 'after', 'equal', 'vague']
    pdtb2_l1_possible_rels = ['Comparison', 'Contingency', 'Expansion', 'Temporal']
    pdtb2_l2_possible_rels = ["Concession", "Contrast", "Cause", "Pragmatic cause", "Conjunction", "Instantiation", "Alternative", "List", "Restatement", "Asynchronous", "Synchrony"]
    pdtb3_l2_possible_rels = ['Purpose', 'Concession', 'Synchronous', 'Asynchronous', 'Contrast', 'Cause', 'Instantiation', 'Substitution', 'Equivalence', 'Manner', 'Condition', 'Conjunction', 'Cause+Belief', 'Level-of-detail']
    if rel_type == "tbd":
        possible_rels = tbd_possible_rels
    elif rel_type == "matres":
        possible_rels = matres_possible_rels
    elif rel_type == "pdtb2_l2":
        possible_rels = pdtb2_l2_possible_rels
    elif rel_type == "pdtb3_l2":
        possible_rels = pdtb3_l2_possible_rels
    else:
        possible_rels = pdtb2_l1_possible_rels
    

    possible_rels = [rel for rel in possible_rels if rel != predicted_rel]

    rel_text = ""
    for po_rel in possible_rels:
        if po_rel in predicted_rel:
            new_rel = random.choice(possible_rels)
            rel_text = predicted_rel.replace(po_rel, new_rel)
    if rel_text == "":
        print(predicted_rel)
        print(rel_type)
        print()
    return rel_text


def init_cyclic_stage_2_pdtb4te_random(stage1_rel_output_file, stage1_conn_rel_output_file, te_data_list, output_dir, te_task, pdtb_type, mode, model_seq):
    #(pdtb_data_list, pdtb_type, te_data_list, te_task, te4pdtb_list, pdtb4te_list, output_dir, mode):
    pdtb4te_rel_data, pdtb4te_conn_rel_data = [], []

    pred_rel_pdtb_data, pred_conn_rel_pdtb_data = [], []
    with open(stage1_rel_output_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())  
            pred_rel_pdtb_data.append(data["predict"])
    
    with open(stage1_conn_rel_output_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())  
            pred_conn_rel_pdtb_data.append(data["predict"])
    
    assert len(te_data_list) == len(pred_rel_pdtb_data)
    assert len(te_data_list) == len(pred_conn_rel_pdtb_data)

    if te_task == "matres":
        label_inst = "The Result MUST be EXACTLY one of: before, after, equal and vague."
        te_info = "before, after, equal and vague"
    else:
        label_inst = "The Result MUST be EXACTLY one of: before, after, is_included, includes, simultaneous and vague."
        te_info = "before, after, is_included, includes, simultaneous and vague"
    

    if "pdtb2_l2" in stage1_rel_output_file:
        random_rel_type = "pdtb2_l2"
    elif "pdtb3_l2" in stage1_rel_output_file:
        random_rel_type = "pdtb3_l2"
    else:
        random_rel_type = "pdtb2_l1"

    for i in range(len(te_data_list)):
            data = te_data_list[i]
            pdtb_pred_rel_info = pred_rel_pdtb_data[i].replace("<think>\n\n</think>\n\n", "")
            pdtb_pred_conn_rel_info = pred_conn_rel_pdtb_data[i].replace("<think>\n\n</think>\n\n", "")

            pdtb_pred_rel_info = random_replace_predicted_rel(pdtb_pred_rel_info, random_rel_type)

            tagged_e1 = data["tagged_e1"]
            tagged_e2 = data["tagged_e2"]
            e1_sent_id, e2_sent_id = data["event1_sent_id"], data["event2_sent_id"]
            e1_sent, e2_sent = data["event1_sentence"], data["event2_sentence"]
            context_sents = data['context_sents']

            relation = data["relation"].lower()

            #context_sents = data["context_sents"]
            conn = None

            relation = data["relation"].lower().strip()
            rel = relation

            if e1_sent_id != e2_sent_id:
                arg1 = e1_sent
                arg2 = e2_sent
            else:
                split_res = split_sentence_by_entities(e1_sent)
                [arg1, arg2] = split_res["split_tagged"]
                conn = split_res["connective"]

            rel_instruction = textwrap.dedent(f"""
            Based on the context, your task is to analyze the text of Arg1 and Arg2 and then determine the temporal relation between Event {tagged_e1} and Event {tagged_e2}. You will be provided a predicted discourse relation between Arg1 and Arg2 to help you predict the temporal relation between Event {tagged_e1} and Event {tagged_e2}. You can only refer to it, but not fully trust this prediction. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
            The output format shoud be:
            Relation: [Your predicted relation]""")

            conn_rel_instruction = textwrap.dedent(f"""
            Based on the context, your task is to analyze the text of Arg1 and Arg2 and then determine the temporal relation between Event {tagged_e1} and Event {tagged_e2}. You will be provided a predicted connective and a discourse relation between Arg1 and Arg2 to help you predict the temporal relation between Event {tagged_e1} and Event {tagged_e2}. You can only refer to it, but not fully trust this prediction. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
            The output format shoud be:
            Relation: [Your predicted relation]""")

            input_text = textwrap.dedent(f"""Input text:
            Context: {context_sents}
            Arg1: {arg1}
            Arg2: {arg2}
            Predicted Discourse {pdtb_pred_rel_info}  
            """)

            conn_input_text = textwrap.dedent(f"""Input text:
            Context: {context_sents}
            Arg1: {arg1}
            Arg2: {arg2}
            Predicted Connective and Discourse Relation: {pdtb_pred_conn_rel_info}
            """)

            #rel_text = f"Relation: {relation}"
            rel_text = f"Relation: {rel}"

            rel_input = {
            'instruction':rel_instruction,
            'input':input_text,
            'output':rel_text,
            }
            conn_rel_input = {
            'instruction':conn_rel_instruction,
            'input':conn_input_text,
            'output':rel_text,
            }

            pdtb4te_rel_data.append(rel_input)
            pdtb4te_conn_rel_data.append(conn_rel_input)

    conn_rel_file = output_dir + "cyclic_stage2_conn_rel_{}4{}_{}_{}_random.json".format(pdtb_type, te_task, mode, model_seq)
    rel_file = output_dir + "cyclic_stage2_rel_{}4{}_{}_{}_random.json".format(pdtb_type, te_task, mode, model_seq)

    """with open(conn_rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(pdtb4te_rel_data, json_file, ensure_ascii=False, indent=4)"""

    with open(rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(pdtb4te_conn_rel_data, json_file, ensure_ascii=False, indent=4)
    
    print("File saved in: " + rel_file)
    print("File saved in: " + conn_rel_file)

def init_cyclic_stage_2_te4pdtb_random(stage1_rel_output_file, pdtb_data_list, output_dir, te_task, pdtb_type, mode, model_seq, hq):
    #(pdtb_data_list, pdtb_type, te_data_list, te_task, te4pdtb_list, pdtb4te_list, output_dir, mode):
    te4pdtb_rel_data, te4pdtb_conn_rel_data = [], []
    pred_rel_te_data = []

    if not hq:
        pred_rel_te_data = extract_te_predictions(stage1_rel_output_file)
    else:
        with open(stage1_rel_output_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line.strip())  
                pred_rel_te_data.append(data["predict"])
    
    print(stage1_rel_output_file)
    print(len(pdtb_data_list))
    print(len(pred_rel_te_data))
    assert len(pdtb_data_list) == len(pred_rel_te_data)

    if "tbd" in stage1_rel_output_file:
        random_rel_type = "tbd"
    elif "matres" in stage1_rel_output_file:
        random_rel_type = "matres"
    else:
        raise RuntimeError("type name error.")

    if "l1" in pdtb_type:
        label_set = L1_LABELS
    elif pdtb_type == "pdtb2_l2":
        label_set = PDTB2_L2_LABELS
    else:
        label_set = PDTB3_L2_LABELS
        
    for i in range(len(pdtb_data_list)):
        pred_te = pred_rel_te_data[i]

        data = pdtb_data_list[i]
        arg1 = data["arg1"]
        arg2 = data["arg2"]
        conn = data['conn']
        conn_list = conn.split("##")
        rel = data["rel"]

        pred_te = random_replace_predicted_rel(pred_te, random_rel_type)

        conn_rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2. First, identify a potential connective that fits the gap. Then, determine the discourse relation label. You will be provided a predicted temporal relation between events in Arg1 and Arg2 to help you predict the discourse relation. You can only refer to it, but not fully trust this prediction.
        Output your reasoning about the potential connectiveand the discourse relation. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Connective: [Your predicted connective], Relation: [Your predicted relation]""")

        rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2 and determine the discourse relation label. You will be provided a predicted temporal relation between events in Arg1 and Arg2 to help you predict the discourse relation. You can only refer to it, but not fully trust this prediction. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Relation: [Your predicted relation]""")


        input_text = textwrap.dedent(f"""Input text:
        Arg1: {arg1}
        Arg2: {arg2}

        Predicted temporal relation: {pred_te}
        """)

        conn_rel_text = f"Connective: {conn_list[0]}, Relation: {rel}"

        rel_text = f"Relation: {rel}"

        conn_rel_input = {
            'instruction':conn_rel_instruction,
            'input':input_text,
            'output':conn_rel_text,
        }

        rel_input = {
            'instruction':rel_instruction,
            'input':input_text,
            'output':rel_text,
        }

        te4pdtb_conn_rel_data.append(conn_rel_input)
        te4pdtb_rel_data.append(rel_input)


    conn_rel_file = output_dir + "cyclic_stage2_conn_rel_{}4{}_{}_{}_random.json".format(te_task, pdtb_type, mode, model_seq)
    rel_file = output_dir + "cyclic_stage2_rel_{}4{}_{}_{}_random.json".format(te_task, pdtb_type, mode, model_seq)

    with open(conn_rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(te4pdtb_conn_rel_data, json_file, ensure_ascii=False, indent=4)

    with open(rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(te4pdtb_rel_data, json_file, ensure_ascii=False, indent=4)

    print("File saved in: " + rel_file)
    print("File saved in: " + conn_rel_file)

def init_stage2_data_random_preprocess():
    te_tasks = ["matres", "tbd"]
    model_seq = [1, 2]
    pdtb_types = ['pdtb2_l1', 'pdtb2_l2', 'pdtb3_l1', 'pdtb3_l2']
    modes = ["train", "test"]
    output_dir = LLAMA_FACTORY_DIR + "/cyclic_rebutall_data/"
    for te in te_tasks:
        for pt in pdtb_types:
            for mode in modes:
                for mos in model_seq:

                    #stage1_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/cyclic_stage1_rel_{}_{}_{}_{}/generated_predictions.jsonl".format(te, pt_f, mode, mos)
                    stage1_conn_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res_stage2/dym_conn_rel_{}_{}_{}_{}/generated_predictions.jsonl".format(te, pt, mode, mos)

                    file_name = "./data/te_data/{}_{}_data.json".format(te, mode)
                    with open(file_name, 'r', encoding='utf-8') as f:
                        te_data_list = json.load(f)

                    init_cyclic_stage_2_pdtb4te_random(stage1_conn_rel_output_file, stage1_conn_rel_output_file, te_data_list, output_dir, te, pt, mode, mos)

                    stage1_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res_stage2/dym_conn_rel_{}_{}_{}_{}/generated_predictions.jsonl".format(pt, te, mode, mos)
                    file_name = "./data/pdtb_data/implicit_{}_{}.json".format(pt, mode)

                    with open(file_name, 'r', encoding='utf-8') as f:
                        pdtb_data_list = json.load(f)

                    init_cyclic_stage_2_te4pdtb_random(stage1_rel_output_file, pdtb_data_list, output_dir, te, pt, mode, mos, False)

                #hq_file = LLAMA_FACTORY_DIR + "/saves/Llama-3.3-70B-Instruct/te4pdtb_res/{}_{}_sup_{}/generated_predictions.jsonl".format(pt_f, te, mode)

                #init_cyclic_stage_2_te4pdtb(hq_file, pdtb_data_list, output_dir+"hq", te, pt, mode, mos2, True)

from datasets import load_dataset


if __name__ == "__main__":
    #init_data_preprocess()
    #init_stage2_data_preprocess()
    #init_stage2_data_gen_preprocess()
    #sinit_stage2_data_random_preprocess()
    pass
