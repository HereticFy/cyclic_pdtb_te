import os
import json
import random

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

def get_list_dict_for_task(task_name, level):
    if task_name.lower() == "pdtb2":
        if level == 1:
            return task_relations["pdtb2_1"], pdtb2_l1_label2idx
        else:
            return task_relations["pdtb2_2"], pdtb2_l2_label2idx
    elif task_name.lower() == "pdtb3":
        if level == 1:
            return task_relations["pdtb3_1"], pdtb3_l1_label2idx
        else:
            return task_relations["pdtb3_2"], pdtb3_l2_label2idx


def write_to_file_json(file_path, data_list, json_type=1):
    """
    write a list of dict obj into file
    Args
        file_path: the path of the file
        data_list: a list of dict data
        json_type:
            1. write with a format same to .json file
            2. write with a format same to .jsonl file. Each line a dict obj
    """
    if json_type == 1:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            for data in data_list:
                f.write("%s\n"%(json.dumps(data, ensure_ascii=False)))


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

def convert_json_format(file_path, mode="1to2"):
    """
    conversion between json_type=1 and json_type=2
    Args:
        file_path:
        mode: 1to2
              2to2
    """
    assert mode in ["1to2", "2to1"]

    new_file_path = file_path.replace(".json", "-{}.json".format(mode))
    if mode == "1to2":
        data_list = read_from_file_json(file_path, json_type=1)
        write_to_file_json(new_file_path, data_list, json_type=2)
    elif mode == "2to1":
        data_list = read_from_file_json(file_path, json_type=2)
        write_to_file_json(new_file_path, json_type=1)

def read_prompt_file(prompt_file):
    """
    read prompt from file
    """
    with open(prompt_file, 'r', encoding="utf-8") as file:
        return file.read()


def data_to_alpaca_format(
    data_list,
    level,
    rel_list,
    instruction_template,
    response_template
):
    new_data_list = []
    valid_idx = 1
    for data in data_list:
        relation_type = data['relation_type']
        if relation_type.lower() != 'implicit':
            continue

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

        instruction = instruction_template.render(relations=", ".join(rel_list), arg1=arg1, arg2=arg2)
        response = response_template.render(relation=rel)

        sample = {
            'id': f'{valid_idx}',
            'system_prompt': 'You are an AI assistant tasked with implicit discourse relation classification.',
            'instruction': instruction,
            'output': response
        }
        new_data_list.append(sample)
        valid_idx += 1

    return new_data_list


def data_to_general_format(
    data_list,
    level,
    rel_list,
):
    all_samples, implicit_samples, explicit_samples = [], [], []
    valid_idx = 1
    for data in data_list:
        relation_type = data['relation_type']
        if relation_type.lower() != 'implicit' or relation_type.lower() != 'explicit':
            continue

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
        }
        all_samples.append(sample)
        if relation_type.lower() == 'implicit':
            implicit_samples.append(sample)
        if relation_type.lower() == 'explicit':
            explicit_samples.append(sample)
        valid_idx += 1

    return all_samples, implicit_samples, explicit_samples