"""Shared code for Stage 1 (stage1_multitask.py) and Stage 2 (stage2_cross_task.py):
label sets, input construction, the prompts of Figures 2-3 of the paper, chat formatting,
generation and loss computation."""
import json
import os
import random
import re
import textwrap

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Label sets
# ---------------------------------------------------------------------------

L1_LABELS = "Temporal, Contingency, Comparison, or Expansion"
PDTB2_L2_LABELS = "Concession, Contrast, Cause, Pragmatic cause, Conjunction, Instantiation, Alternative, List, Restatement, Asynchronous or Synchrony"
PDTB3_L2_LABELS = "Concession, Contrast, Cause, Cause+Belief, Condition, Purpose, Conjunction, Equivalence, Instantiation, Level-of-detail, Manner, Substitution, Asynchronous or Synchronous"

TE_LABELS = {
    "matres": "before, after, equal and vague",
    "tbd": "before, after, is_included, includes, simultaneous and vague",
}


def pdtb_label_set(pdtb_type):
    if "l1" in pdtb_type:
        return L1_LABELS
    return PDTB2_L2_LABELS if pdtb_type == "pdtb2_l2" else PDTB3_L2_LABELS


def te_label_info(te_task):
    """(label instruction, label string) for the full TRE label set of the dataset."""
    labels = TE_LABELS[te_task]
    return f"The Result MUST be EXACTLY one of: {labels}.", labels


def label_list(label_string):
    """'A, B, or C' / 'A, B and C' -> ['A', 'B', 'C']"""
    return [x.strip() for x in re.split(r",\s*(?:or\s+|and\s+)?|\s+or\s+|\s+and\s+", label_string) if x.strip()]


# ---------------------------------------------------------------------------
# Splitting a sentence that contains both events into Arg1 / Arg2 (paper, Appendix F)
# ---------------------------------------------------------------------------

def extract_entities(text):
    """Extract tagged entities and their positions."""
    pattern = re.compile(r"<([a-zA-Z]+\d+)>(.*?)</\1>")
    entities = []
    for match in pattern.finditer(text):
        full_tagged = match.group(0)
        pure_text = match.group(2)
        start = match.start()
        end = match.end()
        entities.append((pure_text, start, end, full_tagged))
    return entities

def find_next_space(text, start_idx):
    """Find the next whitespace so that words are not cut."""
    while start_idx < len(text) and text[start_idx] not in [" ", "\n"]:
        start_idx += 1
    return start_idx

def remove_tags(text):
    """Remove all entity tags to produce clean text."""
    return re.sub(r"</?[a-zA-Z]+\d+>", "", text)



def split_sentence_by_entities(text):
    """
    Split a sentence into two arguments around its two tagged entities.
    - the two entities must end up in different arguments
    - split preferably at a connective, then a relation word, then a comma
    - otherwise fall back to the first whitespace after the first entity
    """

    """pdtb_connectives = [
        "because", "although", "since", "after", "before", "when", "while", "whereas", "though", "if",
        "even though", "even if", "unless", "until", "once", "as", "so that", "now that", "in case",
        "provided that", "due to", "as a result", "therefore", "hence", "thus", "nevertheless", "nonetheless",
        "however", "despite", "in spite of", "albeit", "regardless of", "notwithstanding", "so", "accordingly",
        "for example", "for instance", "consequently", "as long as", "assuming that", "in order to",
        "rather than", "whether or not", "except that", "even as", "lest", "inasmuch as", "thereafter",
        "whereupon", "wherein", "therein", "otherwise"
    ]"""
    pdtb_connectives = ['in addition', 'afterwards', 'by doing so', 'if you were', 'while being', 'subsequently', 'in the end', 'further', 'whereas', 
    'by', 'if it was', 'while', 'after all', 'overall', 'first', 'if one is to', 'moreever', 'eventually', 'simultaneously', 'considering that', 
    'specifically', 'all the while', 'more to the point', 'at the time', 'for the reason that', 'given that', 'generally', 'for that purpose', 
    'as a matter of fact', 'meanwhile', 'after being', 'conversely', 'despite being', 'so far', 'moreover', 'indeed', 'previously', 'given', 'because of',
    'if I am', 'next', 'consequently', 'if we are', 'likewise', 'for instance', 'separately', 'in order', 'although', 'for one', 'if it is', 'in return', 
    'alternatively', 'at the same time', 'as it turns out', 'in summary', 'in response', 'This is because', 'in that', 'during that time', 'for the purpose of',
     'by means of', 'if they were', 'in order to be', 'by comparison', 'through', 'in contrast', 'with', 'rather', 'as a result of having', 'so that',
      'by contrast', 'in comparison', 'still', 'after having', 'plus', 'for', 'on the whole', 'in particular', 'besides', 'inasmuch as', 'in', 'prior to this',
       'as of now', 'in order to', 'as part of that', 'insofar as', 'it is because', 'as a result of being', 'even though', 'if', 'ultimately', 'later', 
       'in order (for them)', 'as', 'for one thing', 'as a result of', 'in short', 'in sum', 'if it were', 'in addition to being', 'also', 'furthermore', 
       'at that time', 'namely', 'if he is', 'if she is', 'upon being', 'if we were', 'ever since', 'by being', 'were one to', 'in general', 'before that', 
       "what's more", 'in addition to', 'since then', 'nonetheless', 'in comparison to the fact', 'similarly', 'to this end', 'in this case', 'in fact', 
       'on the one hand', 'on the contrary', 'though', 'thus being', 'however', 'as a consequence', 'so', 'but', 'soon', 'thereafter', 'earlier', 'as such', 
       'if one is', 'when', 'with the purpose of', 'in other words', 'additionally', 'therefore', 'second', 'in turn', 'if they are', 'granted', 'thereby', 
       'as a result', 'and', 'on account of being', 'because', 'if he is', 'in being', 'that is', 'despite this', 'after', 'if she is', 'hence', 'despite', 
       'for example', 'with the goal', 'then', 'since', 'in the meantime', 'because it was', 'nevertheless', 'incidentally', 'for that reason', 
       'with the goal of', 'as evidence', 'regardless', 'in more detail', 'thus', 'accordingly', 'upon', 'on the other hand', 'if you are', 'if one were', 
       'instead', 'finally', 'otherwise', 'so as', 'yet', 'now', 'before', 'because of that', 'or', 'particularly', 'because of being', 'third', 
       'more specifically']

    relation_words = ["that", "which", "where", "who", "when", "whose"]

    entities = extract_entities(text)
    if len(entities) != 2:
        return {"error": f"Expected 2 entities, found {len(entities)}; cannot split."}

    entities = sorted(entities, key=lambda x: x[1])
    entity1_text, e1_start, e1_end, entity1_full = entities[0]
    entity2_text, e2_start, e2_end, entity2_full = entities[1]

    between_text = text[e1_end:e2_start]

    cut_point = None
    found_type = None
    conn_found = None

    for conn in sorted(pdtb_connectives, key=lambda x: -len(x)):
        pattern_conn = re.compile(rf"\b{re.escape(conn)}\b", flags=re.IGNORECASE)
        match = pattern_conn.search(between_text.lower())
        if match:
            cut_point = e1_end + match.start()
            found_type = f"PDTB_Connective: {conn}"
            conn_found = conn
            break

    if cut_point is None:
        for rel in relation_words:
            pattern_rel = re.compile(rf"\b{re.escape(rel)}\b", flags=re.IGNORECASE)
            match = pattern_rel.search(between_text)
            if match:
                cut_point = e1_end + match.start()
                found_type = f"Relation_Word: {rel}"
                break

    if cut_point is None:
        comma_idx = between_text.find(',')
        if comma_idx != -1:
            cut_point = e1_end + comma_idx
            found_type = "Comma"

    if cut_point is None:
        cut_point = find_next_space(text, e1_end)
        found_type = "Space_Fallback"

    arg1_tagged = text[:cut_point].strip()
    arg2_tagged = text[cut_point:].strip()

    arg1_clean = remove_tags(arg1_tagged).strip()
    arg2_clean = remove_tags(arg2_tagged).strip()

    return {
        "split_tagged": [arg1_tagged, arg2_tagged],
        "connective": conn_found,
        "split_clean": [arg1_clean, arg2_clean],
        "note": f"Split by: {found_type}, after character {cut_point}"
    }





# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def prepare_te_sample(data):
    """TRE sample as Arg1 / Arg2 with the two tagged events and the original (gold) label."""
    if data["event1_sent_id"] != data["event2_sent_id"]:
        arg1, arg2 = data["event1_sentence"], data["event2_sentence"]
    else:
        arg1, arg2 = split_sentence_by_entities(data["event1_sentence"])["split_tagged"]
    return {
        "arg1": arg1, "arg2": arg2, "tagged_e1": data["tagged_e1"], "tagged_e2": data["tagged_e2"],
        "label": data["relation"].lower().strip(),
    }


def prepare_pdtb_sample(data):
    return {
        "arg1": data["arg1"], "arg2": data["arg2"],
        "conn": data["conn"].split("##")[0], "label": data["rel"],
    }


def load_data(data_dir, te_task, pdtb_type, split, max_samples=None):
    """Return (TRE samples, IDRP samples) of one split from data/te_data and data/pdtb_data."""
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[:max_samples] if max_samples else data
    te = [prepare_te_sample(d) for d in load(os.path.join(data_dir, "te_data", f"{te_task}_{split}_data.json"))]
    dr = [prepare_pdtb_sample(d) for d in load(os.path.join(data_dir, "pdtb_data", f"implicit_{pdtb_type}_{split}.json"))]
    return te, dr


# ---------------------------------------------------------------------------
# Prompts of Stage 1 (paper, Figures 2 and 3)
# ---------------------------------------------------------------------------

def te_prompt(s, te_task):
    """TRE prompt (Figure 2). Returns (prompt, target)."""
    label_inst, _ = te_label_info(te_task)
    prompt = textwrap.dedent(f"""
    Your task is to analyze the text of Arg1 and Arg2 and then determine the temporal relation between Event {{e1}} and Event {{e2}}. {label_inst} Note that a VAGUE label means the temporal relation is ambiguous; when you cannot find any evidence for other temporal relations, you can predict the relation as vague.
    The output format should be:
    Relation: [Your predicted relation]
    Input text:
    Arg1: {{arg1}}
    Arg2: {{arg2}}""").strip().format(e1=s["tagged_e1"], e2=s["tagged_e2"], arg1=s["arg1"], arg2=s["arg2"])
    return prompt, f"Relation: {s['label']}"


def discourse_prompt(arg1, arg2, pdtb_type):
    """IDRP prompt (Figure 3); the same prompt generates auxiliary discourse predictions for TRE inputs (Figure 4)."""
    return textwrap.dedent(f"""
    Your task is to analyze the discourse relation between Arg1 and Arg2. First, identify a potential connective that fits the gap. Then, determine the discourse relation label.
    Output your reasoning about the potential connective and the discourse relation. The relation MUST be EXACTLY one of: {pdtb_label_set(pdtb_type)}.
    The output format should be:
    Connective: [Your predicted connective], Relation: [Your predicted relation]
    Input text:
    Arg1: {{arg1}}
    Arg2: {{arg2}}""").strip().format(arg1=arg1, arg2=arg2)


def pdtb_prompt(s, pdtb_type):
    """IDRP prompt (Figure 3). Returns (prompt, target)."""
    return discourse_prompt(s["arg1"], s["arg2"], pdtb_type), f"Connective: {s['conn']}, Relation: {s['label']}"


# ---------------------------------------------------------------------------
# Chat formatting, generation, loss
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_generation(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class ChatFormatter:
    """Chat formatting (Llama-3 format, as LLaMA-Factory's `llama3` template; other models use their chat template)."""

    def __init__(self, tokenizer, cutoff_len):
        self.tok = tokenizer
        self.cutoff_len = cutoff_len
        self.is_llama3 = "<|start_header_id|>" in tokenizer.get_vocab()

    def prompt_text(self, conversation):
        """`conversation` is a user message (str) or a list of alternating user/assistant messages
        that ends with a user message."""
        if isinstance(conversation, str):
            conversation = [conversation]
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": m} for i, m in enumerate(conversation)]
        if self.is_llama3:
            text = "<|begin_of_text|>"
            for m in messages:
                text += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
            return text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def response_text(self, response):
        return response + (self.tok.eos_token if not self.is_llama3 else "<|eot_id|>")

    def encode_train(self, user_msg, response):
        """Return input_ids and labels; the loss is computed on the response only."""
        p_ids = self.tok.encode(self.prompt_text(user_msg), add_special_tokens=False)
        r_ids = self.tok.encode(self.response_text(response), add_special_tokens=False)
        max_prompt = max(self.cutoff_len - len(r_ids), 1)
        p_ids = p_ids[-max_prompt:] if len(p_ids) > max_prompt else p_ids
        input_ids = (p_ids + r_ids)[: self.cutoff_len]
        labels = ([-100] * len(p_ids) + r_ids)[: self.cutoff_len]
        return input_ids, labels

    def collate_train(self, encoded, device):
        max_len = max(len(ids) for ids, _ in encoded)
        pad = self.tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for ids, lab in encoded:
            n = max_len - len(ids)
            input_ids.append(ids + [pad] * n)
            labels.append(lab + [-100] * n)
            attn.append([1] * len(ids) + [0] * n)
        to = lambda x: torch.tensor(x, dtype=torch.long, device=device)
        return to(input_ids), to(attn), to(labels)


@torch.no_grad()
def generate(model, fmt, prompts, batch_size, max_new_tokens, temperature, top_p, desc=None):
    """Frozen inference (no parameter update). Returns the decoded continuations."""
    was_training = model.training
    model.eval()
    tok = fmt.tok
    tok.padding_side = "left"
    stop_ids = [tok.eos_token_id]
    if fmt.is_llama3:
        stop_ids.append(tok.convert_tokens_to_ids("<|eot_id|>"))

    outputs = []
    batches = range(0, len(prompts), batch_size)
    for i in (tqdm(batches, desc=desc) if desc else batches):
        texts = [fmt.prompt_text(p) for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                  truncation=True, max_length=fmt.cutoff_len).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=stop_ids,
            use_cache=True,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))

    tok.padding_side = "right"
    if was_training:
        model.train()
    return outputs


def backward_on_batch(model, fmt, examples, micro_batch_size):
    """Compute the loss of one task batch (split into micro-batches) and back-propagate it.
    The gradients are accumulated; the caller decides when to update the parameters."""
    model.train()
    micro_batches = [examples[i:i + micro_batch_size] for i in range(0, len(examples), micro_batch_size)]
    total = 0.0
    for chunk in micro_batches:
        encoded = [fmt.encode_train(p, r) for p, r in chunk]
        input_ids, attn, labels = fmt.collate_train(encoded, model.device)
        loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
        (loss / len(micro_batches)).backward()
        total += loss.item() / len(micro_batches)
    return total


def optimizer_step(model, optimizer, scheduler, max_grad_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def make_batches(samples, batch_size, rng):
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    return [[samples[i] for i in idx[j:j + batch_size]] for j in range(0, len(idx), batch_size)]


def write_predictions(path, rows):
    """Write predictions in LLaMA-Factory's generated_predictions.jsonl format (+ optional extra fields)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
