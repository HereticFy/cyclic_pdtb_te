from gum_preprocessing import *
from data_preprocess import *
from paths import LLAMA_FACTORY_DIR

def preprocess_gum_init(gum_data_list, output_dir, mode):
    gum_rel_data = []
    label_set = ['causal', 'concession', 'conjunction', 'purpose', 'reformulation', 'contrast', 'query', 'condition', 'mode', 'alternation', 'elaboration', 'temporal', 'explanation']
    for i in range(len(gum_data_list)):
        data = gum_data_list[i]
        arg1 = data["arg1"]
        arg2 = data["arg2"]

        rel = data["rel"]

        rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2 and determine the discourse relation label. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Relation: [Your predicted relation]""")

        input_text = textwrap.dedent(f"""Input text:
        Arg1: {arg1}
        Arg2: {arg2}""")

        rel_text = f"Relation: {rel}"

        rel_input = {
            'instruction':rel_instruction,
            'input':input_text,
            'output':rel_text,
        }

        gum_rel_data.append(rel_input)

    rel_file = output_dir + "eng.pdtb.gum_rel_{}.json".format(mode)

    with open(rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(gum_rel_data, json_file, ensure_ascii=False, indent=4)



def init_gum_preprocess():
    output_dir = LLAMA_FACTORY_DIR + "/cyclic_gum_data/"
    modes = ["train", "test"]
    for mode in modes:
        file_name = "./data/disrpt_data/eng.pdtb.gum_{}.json".format(mode)
        with open(file_name, 'r', encoding='utf-8') as f:
            gum_data_list = json.load(f)

        preprocess_gum_init(gum_data_list, output_dir, mode)


def init_cyclic_gum_stage_1(gum_data_list, te_data_list, te_task, output_dir, mode):
    detailed_te4gum_data, pdtb4te_rel_data = [], []

    label_set = ['causal', 'concession', 'conjunction', 'purpose', 'reformulation', 'contrast', 'query', 'condition', 'mode', 'alternation', 'elaboration', 'temporal', 'explanation']

    if te_task == "matres":
        label_inst = "The Result MUST be EXACTLY one of: before, after, equal and vague."
        te_info = "before, after, equal and vague"
    else:
        label_inst = "The Result MUST be EXACTLY one of: before, after, is_included, includes, simultaneous and vague."
        te_info = "before, after, is_included, includes, simultaneous and vague"

    for i in range(len(gum_data_list)):
        data = gum_data_list[i]
        arg1 = data["arg1"]
        arg2 = data["arg2"]

        rel = data["rel"]


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


        de_te4gum_input = {
            'instruction':detailed_instruction,
            'input':input_text,
            'output':rel,
        }

        detailed_te4gum_data.append(de_te4gum_input)

    for i in range(len(te_data_list)):
        data = te_data_list[i]

        tagged_e1 = data["tagged_e1"]
        tagged_e2 = data["tagged_e2"]
        e1_sent_id, e2_sent_id = data["event1_sent_id"], data["event2_sent_id"]
        e1_sent, e2_sent = data["event1_sentence"], data["event2_sentence"]

        relation = data["relation"].lower().strip()
        rel = relation

        if e1_sent_id != e2_sent_id:
            arg1 = e1_sent
            arg2 = e2_sent
        else:
            split_res = split_sentence_by_entities(e1_sent)
            [arg1, arg2] = split_res["split_tagged"]

        gum_rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2 and determine the discourse relation label. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Relation: [Your predicted relation]""")

        input_text = textwrap.dedent(f"""Input text:
            Arg1: {arg1}
            Arg2: {arg2}""")


        pdtb4te_rel_input = {
            'instruction':gum_rel_instruction,
            'input':input_text,
            'output':rel,
        }

        pdtb4te_rel_data.append(pdtb4te_rel_input)

    de_te4gum_file = output_dir + "cyclic_stage1_detailed_gum_{}_{}.json".format(te_task, mode)
    pdtb4te_rel_file = output_dir + "cyclic_stage1_rel_{}_gum_{}.json".format(te_task, mode)

    with open(de_te4gum_file, 'w', encoding='utf-8') as json_file:
        json.dump(detailed_te4gum_data, json_file, ensure_ascii=False, indent=4)
    
    with open(pdtb4te_rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(pdtb4te_rel_data, json_file, ensure_ascii=False, indent=4)

    print("File saved in: " + de_te4gum_file)
    print("Data for cyclic stage1 is saved!")



def init_gum_stage1_preprocess():
    output_dir = LLAMA_FACTORY_DIR + "/cyclic_gum_data/"
    modes = ["train", "test"]
    te_tasks = ["matres", "tbd"]
    for mode in modes:
        for te_task in te_tasks:
            file_name = "./data/te_data/{}_{}_data.json".format(te_task, mode)
            with open(file_name, 'r', encoding='utf-8') as f:
                te_data_list = json.load(f)

            file_name = "./data/disrpt_data/eng.pdtb.gum_{}.json".format(mode)
            with open(file_name, 'r', encoding='utf-8') as f:
                gum_data_list = json.load(f)

            init_cyclic_gum_stage_1(gum_data_list, te_data_list, te_task, output_dir, mode)


def init_cyclic_stage_2_pdtb4te(stage1_rel_output_file, te_data_list, output_dir, te_task, mode, model_seq):
    #(gum_data_list, pdtb_type, te_data_list, te_task, te4pdtb_list, pdtb4te_list, output_dir, mode):
    pdtb4te_rel_data = []

    pred_rel_pdtb_data = []
    with open(stage1_rel_output_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            pred_rel_pdtb_data.append(data["predict"])

    assert len(te_data_list) == len(pred_rel_pdtb_data)

    if te_task == "matres":
        label_inst = "The Result MUST be EXACTLY one of: before, after, equal and vague."
        te_info = "before, after, equal and vague"
    else:
        label_inst = "The Result MUST be EXACTLY one of: before, after, is_included, includes, simultaneous and vague."
        te_info = "before, after, is_included, includes, simultaneous and vague"

    for i in range(len(te_data_list)):
            data = te_data_list[i]
            pdtb_pred_rel_info = pred_rel_pdtb_data[i].replace("<think>\n\n</think>\n\n", "")

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



            input_text = textwrap.dedent(f"""Input text:
            Context: {context_sents}
            Arg1: {arg1}
            Arg2: {arg2}
            Predicted Discourse {pdtb_pred_rel_info}  
            """)


            #rel_text = f"Relation: {relation}"
            rel_text = f"Relation: {rel}"

            rel_input = {
            'instruction':rel_instruction,
            'input':input_text,
            'output':rel_text,
            }

            pdtb4te_rel_data.append(rel_input)

    rel_file = output_dir + "cyclic_stage2_rel_gum4{}_{}_{}.json".format(te_task, mode, model_seq)

    with open(rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(pdtb4te_rel_data, json_file, ensure_ascii=False, indent=4)
    
    print("File saved in: " + rel_file)



def init_cyclic_stage_2_te4pdtb(stage1_rel_output_file, gum_data_list, output_dir, te_task, mode, model_seq, hq):
    #(gum_data_list, pdtb_type, te_data_list, te_task, te4pdtb_list, pdtb4te_list, output_dir, mode):
    te4pdtb_rel_data = []
    pred_rel_te_data = []

    label_set = ['causal', 'concession', 'conjunction', 'purpose', 'reformulation', 'contrast', 'query', 'condition', 'mode', 'alternation', 'elaboration', 'temporal', 'explanation']

    if not hq:
        pred_rel_te_data = extract_te_predictions(stage1_rel_output_file)
    else:
        with open(stage1_rel_output_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line.strip())  
                pred_rel_te_data.append(data["predict"])
    
    print(stage1_rel_output_file)
    print(len(gum_data_list))
    print(len(pred_rel_te_data))
    assert len(gum_data_list) == len(pred_rel_te_data)

        
    for i in range(len(gum_data_list)):
        pred_te = pred_rel_te_data[i]

        data = gum_data_list[i]
        arg1 = data["arg1"]
        arg2 = data["arg2"]
        rel = data["rel"]

        rel_instruction = textwrap.dedent(f"""
        Your task is to analyze the discourse relation between Arg1 and Arg2 and determine the discourse relation label. You will be provided a predicted temporal relation between events in Arg1 and Arg2 to help you predict the discourse relation. You can only refer to it, but not fully trust this prediction. The relation MUST be EXACTLY one of: {label_set}.
        The output format shoud be:
        Relation: [Your predicted relation]""")


        input_text = textwrap.dedent(f"""Input text:
        Arg1: {arg1}
        Arg2: {arg2}

        Predicted temporal relation: {pred_te}
        """)

        rel_text = f"Relation: {rel}"

        rel_input = {
            'instruction':rel_instruction,
            'input':input_text,
            'output':rel_text,
        }

        te4pdtb_rel_data.append(rel_input)


    rel_file = output_dir + "cyclic_stage2_rel_{}4gum_{}_{}.json".format(te_task, mode, model_seq)

    with open(rel_file, 'w', encoding='utf-8') as json_file:
        json.dump(te4pdtb_rel_data, json_file, ensure_ascii=False, indent=4)

    print("File saved in: " + rel_file)


def init_stage2_data_preprocess():
    te_tasks = ["matres", "tbd"]
    model_seq = [1, 2]
    modes = ["train", "test"]
    output_dir = LLAMA_FACTORY_DIR + "/cyclic_gum_data/"
    for te in te_tasks:
        for mode in modes:
            for mos in model_seq:                
                stage1_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/cyclic_stage1_rel_{}_gum_{}_{}/generated_predictions.jsonl".format(te, mode, mos)

                file_name = "./data/te_data/{}_{}_data.json".format(te, mode)
                with open(file_name, 'r', encoding='utf-8') as f:
                    te_data_list = json.load(f)

                init_cyclic_stage_2_pdtb4te(stage1_rel_output_file, te_data_list, output_dir, te, mode, mos)

            for mos2 in model_seq:
                stage1_rel_output_file = LLAMA_FACTORY_DIR + "/saves/cyclic_res/cyclic_stage1_detailed_gum_{}_{}_{}/generated_predictions.jsonl".format(te, mode, mos2)

                file_name = "./data/disrpt_data/eng.pdtb.gum_{}.json".format(mode)

                with open(file_name, 'r', encoding='utf-8') as f:
                    pdtb_data_list = json.load(f)

                init_cyclic_stage_2_te4pdtb(stage1_rel_output_file, pdtb_data_list, output_dir, te, mode, mos2, False)

            hq_file = LLAMA_FACTORY_DIR + "/saves/Llama-3.3-70B-Instruct/te4pdtb_res/pdtb_gum_{}_sup_{}/generated_predictions.jsonl".format(te, mode)
            init_cyclic_stage_2_te4pdtb(hq_file, pdtb_data_list, output_dir+"hq", te, mode, mos2, True)



if __name__ == "__main__":
    #init_gum_preprocess()
    #init_gum_stage1_preprocess()
    #init_stage2_data_preprocess()
    pass
