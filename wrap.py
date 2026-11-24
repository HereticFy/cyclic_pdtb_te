import os
import json
from jinja2 import Template

from utils import read_from_file_json, task_relations, write_to_file_json, data_to_alpaca_format

def file_to_alpaca_format(
    task_name,
    mode,
    rel_level,
    instruction_template_file,
    response_template_file
):
    raw_data_file = "data/raw/{}/{}.json".format(task_name, mode)
    data_list = read_from_file_json(raw_data_file, json_type=2)
    relations = task_relations["{}_{}".format(task_name, rel_level)]

    instruction_name = instruction_template_file.split(".")[0]
    instruction_template_path = "data/templates/{}".format(instruction_template_file)
    response_template_path = "data/templates/{}".format(response_template_file)
    instruction_template = Template(open(instruction_template_path, 'r').read())
    response_template = Template(open(response_template_path, 'r').read())

    alpaca_data_list = data_to_alpaca_format(
        data_list, rel_level, relations,
        instruction_template, response_template
    )
    out_dir = "data/wrapped/{}/l{}/{}".format(task_name, rel_level, instruction_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "{}.json".format(mode))
    write_to_file_json(out_file, alpaca_data_list)


def file_to_general_format(
    task_name,
    mode,
    rel_level,
    instruction_template_file,
    response_template_file
):
    pass
    """raw_data_file = "data/raw/{}/{}.json".format(task_name, mode)
    data_list = read_from_file_json(raw_data_file, json_type=2)
    relations = task_relations["{}_{}".format(task_name, rel_level)]

    instruction_name = instruction_template_file.split(".")[0]
    instruction_template_path = "data/templates/{}".format(instruction_template_file)
    response_template_path = "data/templates/{}".format(response_template_file)
    instruction_template = Template(open(instruction_template_path, 'r').read())
    response_template = Template(open(response_template_path, 'r').read())

    alpaca_data_list = data_to_alpaca_format(
        data_list, rel_level, relations,
        instruction_template, response_template
    )
    out_dir = "data/wrapped/{}/l{}/{}".format(task_name, rel_level, instruction_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "{}.json".format(mode))
    write_to_file_json(out_file, alpaca_data_list)"""

if __name__ == "__main__":
    task_name = "pdtb3"
    mode = "train"
    level = 2
    instruction_template_file = "plain.txt"
    response_template_file = "plain_res.txt"
    for task_name in ["pdtb2", "pdtb3"]:
        for mode in ["train", "dev", "test"]:
            for level in [1, 2]:
                file_to_alpaca_format(
                    task_name, mode, level,
                    instruction_template_file,
                    response_template_file
                )