import os
import json
import random
from paths import PDTB_SOURCE_DIR

task_relations = {
    "pdtb2_1": ["Temporal", "Contingency", "Comparison", "Expansion"],
    "pdtb3_1": ["Temporal", "Contingency", "Comparison", "Expansion"],
    "pdtb2_2": ["Concession", "Contrast", "Cause", "Pragmatic cause", "Conjunction", "Instantiation", "Alternative", "List", "Restatement", "Asynchronous", "Synchrony"],
    "pdtb3_2": ["Concession", "Contrast", "Cause", "Cause+Belief", "Condition", "Purpose", "Conjunction", "Equivalence", "Instantiation", "Level-of-detail", "Manner", "Substitution", "Asynchronous", "Synchronous"]
}

pdtb2_l1_label2idx = {
    "temporal": 0,
    "contingency": 1,
    "comparison": 2,
    "expansion": 3,
}

pdtb2_l2_label2idx = {
    "concession": 0,
    "contrast": 1,
    "cause": 2,
    "pragmatic cause": 3,
    "conjunction": 4,
    "instantiation": 5,
    "alternative" : 6,
    "list": 7,
    "restatement": 8,
    "asynchronous": 9,
    "synchrony": 10
}

pdtb3_l1_label2idx = {
    "temporal": 0,
    "contingency": 1,
    "comparison": 2,
    "expansion": 3,
}

pdtb3_l2_label2idx = {
    "concession": 0,
    "contrast": 1,
    "cause": 2,
    "cause+belief": 3,
    "condition": 4,
    "purpose": 5,
    "conjunction": 6,
    "equivalence": 7,
    "instantiation": 8,
    "level-of-detail": 9,
    "manner": 10,
    "substitution": 11,
    "asynchronous": 12,
    "synchronous": 13
}


def read_from_file_json(file_path, json_type=1):
    """
    Read a list of data from a json file
    Args:
        file_path: file path
        json_type:
            1, data organized in a .json format
            2, data organized in a .jsonl format
    """
    if json_type == 1:
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
            return data_list
    else:
        data_list = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    data_list.append(data)

        return data_list
    

def data_to_general_format(
    data_list,
    level,
    rel_list,
):
    all_samples, implicit_samples, explicit_samples = [], [], []
    valid_idx = 1
    for data in data_list:
        relation_type = data['relation_type']
        """if relation_type.lower() != 'implicit' or relation_type.lower() != 'explicit':
            continue"""

        relation_class = data['relation_class'].split('##')[0]
        relations = relation_class.split('.')
        if len(relations) < level:
            continue

        rel = relations[level-1]
        if rel not in rel_list:
            continue

        conn = data['conn']
        arg1 = data['arg1']
        arg2 = data['arg2']

        sample = {
            "valid_idx": valid_idx,
            "type": relation_type,
            "arg1": arg1,
            "arg2": arg2,
            "conn": conn,
            "rel": rel,
        }
        all_samples.append(sample)
        if relation_type.lower() == 'implicit':
            implicit_samples.append(sample)
        if relation_type.lower() == 'explicit':
            explicit_samples.append(sample)
        valid_idx += 1

    return all_samples, implicit_samples, explicit_samples

if __name__ == "__main__":
    for task_name in ["pdtb2", "pdtb3"]:
        #for mode in ["train", "dev", "test"]:
        for mode in ["train", "dev", "test"]:
            for level in [1, 2]:
                relations = task_relations["{}_{}".format(task_name, level)]
                file_path = PDTB_SOURCE_DIR + "/{}/{}.json".format(task_name, mode)
                data_list = read_from_file_json(file_path, 2)
                all_samples, implicit_samples, explicit_samples = data_to_general_format(data_list, level, relations)
                implict_file = "data/pdtb_data/implicit_{}_l{}_{}.json".format(task_name, level, mode)
                explict_file = "data/pdtb_data/explicit_{}_l{}_{}.json".format(task_name, level, mode)
                with open(implict_file, 'w', encoding='utf-8') as f:
                    json.dump(implicit_samples, f, indent=4, ensure_ascii=False)
                with open(explict_file, 'w', encoding='utf-8') as f:
                    json.dump(explicit_samples, f, indent=4, ensure_ascii=False)