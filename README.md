# Integrating Two Linguistic Phenomena: Bi-directional Interplay between Discourse and Temporal Relations

Code for the paper **"Integrating Two Linguistic Phenomena: Bi-directional Interplay between Discourse and
Temporal Relations"** (Yi Fan and Michael Strube), *Findings of EACL 2027*.

The paper studies the bi-directional interplay between **implicit discourse relation prediction (IDRP)**
on PDTB 2.0 / 3.0 and **temporal relation extraction (TRE)** on MATRES / TimeBank-Dense, with a two-stage
framework: **Stage 1: Multi-task Learning** and **Stage 2: Iterative Cross-task Learning**.
Please refer to the paper for details of the method.

**Please follow the paper to reproduce the method.** In particular, Stage 2 updates the model per batch: for
every batch the model first runs auxiliary inference in a frozen state and is then fine-tuned on the augmented input.
See [Training as in the paper](#training-as-in-the-paper-pytorch--peft) for the corresponding scripts.

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) was only used to run the baselines and to quickly test
the performance of the different settings. The LLaMA-Factory workflow described in [Pipeline](#pipeline) is
therefore a simplified, round-wise approximation of Stage 2 (predict on the whole dataset, build augmented data,
fine-tune) and not the exact training procedure of the paper.

## Setup

```bash
pip install torch transformers peft datasets scikit-learn pandas numpy regex jinja2 tqdm
```

External locations are configured in [`paths.py`](paths.py). Set them via environment variables
(or edit the defaults in the file):

| Variable | Meaning |
|---|---|
| `LLAMA_FACTORY_DIR` | Root of your LLaMA-Factory checkout. Generated data is written here and predictions are read from its `saves/` directory. |
| `PDTB_SOURCE_DIR` | Raw PDTB data, organised as `<pdtb2\|pdtb3>/<train\|dev\|test>.json` (JSON lines). |
| `DISRPT_DIR` | [DISRPT 2025](https://github.com/disrpt/sharedtask2025) data, organised as `<dataset>/<dataset>_<split>.{conllu,tok,rels}`. |
| `DISRPT_TEST_DIR` | DISRPT data directory used for the `eng.pdtb.gum` files. |

```bash
export LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory
export PDTB_SOURCE_DIR=/path/to/pdtb_data
```

Project-internal data lives under `./data/`:

```
data/
├── pdtb_data/   implicit_<pdtb2|pdtb3>_l<1|2>_<split>.json   (created by pdtb_preprocessing.py)
├── te_data/     <matres|tbd>_<train|test>_data.json          (TRE data, see below)
└── disrpt_data/ <dataset>_<split>.json                       (created by gum_preprocessing.py)
```

The TRE data (MATRES / TB-Dense) is preprocessed following [LLMERE](https://github.com/HerbertHu/LLMERE).
Each sample is expected to contain `tagged_e1`, `tagged_e2` (events marked with `<e1>...</e1>`),
`event1_sentence`, `event2_sentence`, `event1_sent_id`, `event2_sent_id`, `context_sents` and `relation`.

## Pipeline

This section describes the data preparation and the LLaMA-Factory workflow used for baselines and quick tests.

Notation used below: `<pdtb>` is one of `pdtb2_l1`, `pdtb2_l2`, `pdtb3_l1`, `pdtb3_l2`; `<te>` is `matres` or `tbd`;
`<split>` is `train` or `test`.
All generated files use LLaMA-Factory's alpaca format (`instruction` / `input` / `output`) and have to be
registered in LLaMA-Factory's `data/dataset_info.json` before they can be used with `--dataset` / `--eval_dataset`.

### 0. Data preprocessing

- `pdtb_preprocessing.py` converts raw PDTB 2.0 / 3.0 (Ji and Eisenstein split) into `data/pdtb_data/implicit_*.json`
  (and `explicit_*.json`) for top-level and second-level senses, following [ConnRel](https://github.com/liuwei1206/ConnRel).
- `gum_preprocessing.py` converts DISRPT `.rels` / `.tok` files (e.g. `eng.pdtb.gum`) into `data/disrpt_data/*.json`.

`data_preprocess.init_data_preprocess()` then writes all single-task and auxiliary-inference inputs to
`$LLAMA_FACTORY_DIR/cyclic_data/`:

| File | Content |
|---|---|
| `rel_<pdtb>_<split>`, `conn_rel_<pdtb>_<split>` | IDRP: predict the discourse relation (and the implicit connective). |
| `rel_<te>_<split>`, `conn_rel_<te>_<split>` | TRE: the event pair is split into Arg1/Arg2; predict the temporal relation (original MATRES / TB-Dense labels). |
| `context_*_<te>_<split>` | TRE with document context. |
| `cyclic_stage1_*` | Inputs for auxiliary inference in Stage 2 (see below). |

### 1. Baselines

- **Zero-shot**: predict directly on the test sets (e.g. `rel_<te>_test`, `rel_<pdtb>_test`).
- **Single-task fine-tuning**: LoRA SFT on one task only (e.g. `rel_<te>_train` or `rel_<pdtb>_train`).

### 2. Stage 1: Multi-task Learning

LoRA SFT on the TRE and the IDRP training sets jointly (`rel_<te>_train` + `conn_rel_<pdtb>_train`), then predict
on both test sets. The resulting adapter is the initialisation for Stage 2. Note that LLaMA-Factory mixes the two
datasets; the alternating-batch training of the paper is implemented in `stage1_multitask.py`.

### 3. Stage 2: Iterative Cross-task Learning (round-wise approximation)

In the paper, auxiliary inference and fine-tuning alternate per batch. In the LLaMA-Factory workflow, each round consists of auxiliary inference with the current (frozen) model, followed by fine-tuning on inputs
augmented with these auxiliary predictions.

**a) Auxiliary inference.** The current adapter predicts (`--do_predict`) on the cross-task inputs created by
`init_cyclic_stage_1()`, on both the train and the test split:

- `cyclic_stage1_rel_<te>_<pdtb>_<split>`, `cyclic_stage1_conn_rel_<te>_<pdtb>_<split>`: TRE samples with the
  IDRP instruction; the model predicts a discourse relation (and connective) for each TRE input.
- `cyclic_stage1_<pdtb>_<te>_<split>`, `cyclic_stage1_detailed_<pdtb>_<te>_<split>`: IDRP samples with a TRE
  instruction; the model extracts event pairs across Arg1/Arg2 and predicts `(Event A, Event B, Relation)` tuples.

**b) Augmented training.** The auxiliary predictions are appended to the original inputs; the target stays the gold
label. The instructions ask the model to treat the auxiliary predictions as reference cues only.

- *Discourse-informed temporal enhancement*: `cyclic_stage2_rel_<pdtb>4<te>_<split>_<k>`, TRE inputs with the
  predicted connective and discourse relation.
- *Temporal-informed discourse enhancement*: `cyclic_stage2_conn_rel_<te>4<pdtb>_<split>_<k>` (and
  `cyclic_stage2_rel_...`), IDRP inputs with the predicted temporal relations.

`<k>` indexes the prediction run the auxiliary features come from. Training continues from the current adapter
on both augmented training sets, followed by prediction on the augmented test sets.

**c) Iteration.**

- First round: `init_stage2_data_preprocess()` reads the auxiliary predictions of the Stage-1 model
  (`$LLAMA_FACTORY_DIR/saves/cyclic_res/cyclic_stage1_*`) and writes the augmented data to
  `$LLAMA_FACTORY_DIR/cyclic_stage2_data/`.
- Further rounds: the updated model predicts the auxiliary features again
  (`$LLAMA_FACTORY_DIR/saves/cyclic_res_stage2/dym_conn_rel_*`), and `init_stage2_data_gen_preprocess()` writes
  new augmented data to `$LLAMA_FACTORY_DIR/cyclic_stage2_gen_data/` (registered with a `gen_` prefix, e.g.
  `gen_cyclic_stage2_conn_rel_<te>4<pdtb>_<split>_<k>` and `gen_cyclic_stage2_rel_<pdtb>4<te>_<split>_<k>`).
  Training continues from the adapter of the previous round.

**Ablations.** `init_stage2_data_random_preprocess()` replaces the auxiliary predictions with random labels
(written to `$LLAMA_FACTORY_DIR/cyclic_rebutall_data/`, suffix `_random`). The "without Stage-2 fine-tuning" setting
predicts on the augmented test sets with the Stage-1 adapter.

### 4. Evaluation

- `eval.py` reads `generated_predictions.jsonl` files (from LLaMA-Factory or from the scripts below) and extracts
  the text after `Relation:`. There must be exactly one prediction per test instance, in the original order.
  - TRE: precision / recall / micro-F1 on MATRES and TB-Dense with their original label sets, excluding `vague`.
    A prediction that is not a valid label is treated like `vague` (no positive prediction; the gold relation counts
    as missed).
  - IDRP: accuracy and macro-F1 with two evaluation methods, E1 (CoNLL 2015 style) and E2 (a correct prediction
    counts for every gold label of a multi-label instance), following [ConnRel](https://github.com/liuwei1206/ConnRel).
    A prediction that is not a valid label is counted as wrong.

  Edit the prediction file paths in its `__main__` block.
- `gum_eval.py`: evaluation on `eng.pdtb.gum`, following the official relation classification scorer of the
  [DISRPT 2025 shared task](https://github.com/disrpt/sharedtask2025) (`utils/disrpt_eval_2024.py`, task `R`):
  accuracy of the `label` column over all relation instances of the test `.rels` file. It writes the predictions as
  a DISRPT `.rels` file and reports the same accuracy as the official scorer, which can also be run directly:

  ```bash
  python gum_eval.py --gum_predictions <generated_predictions.jsonl> \
      [--gold_rels <eng.pdtb.gum_test.rels>] [--te_predictions <TRE predictions> --te_task matres]
  python <sharedtask2025>/utils/disrpt_eval_2024.py -g <eng.pdtb.gum_test.rels> -p <eng.pdtb.gum_test.pred.rels> -t R
  ```

### Domain transfer: eng.pdtb.gum

`input_gum_preprocessing.py` builds the same inputs for `eng.pdtb.gum` from DISRPT 2025
(`init_gum_preprocess`, `init_gum_stage1_preprocess`, `init_stage2_data_preprocess`), writing to
`$LLAMA_FACTORY_DIR/cyclic_gum_data/`.

## Training as in the paper (PyTorch + PEFT)

These scripts train directly with `transformers` + `peft`, without LLaMA-Factory. They read
`data/pdtb_data/` and `data/te_data/` (see [Setup](#setup)).

### Stage 1: Multi-task Learning

`stage1_multitask.py` implements Section 3.1 of the paper. A new LoRA adapter is trained on the TRE and the IDRP
training sets with alternating batches: in every iteration the model processes a TRE batch (prompt of Figure 2,
target `Relation: <temporal relation>`) and an IDRP batch (prompt of Figure 3, target
`Connective: <connective>, Relation: <discourse relation>`). Both losses are back-propagated in the same iteration,
and the parameters are updated only after the gradients of both have been accumulated.

All hyper-parameters have to be set by the user:

```bash
python stage1_multitask.py \
    --model_name_or_path <base model> \
    --pdtb_type pdtb2_l1 --te_task matres \
    --output_dir <Stage-1 output directory> \
    --lora_rank <r> --lora_alpha <alpha> --lora_dropout <dropout> --lora_target_modules <module names> \
    --num_train_epochs <epochs> \
    --per_device_train_batch_size <batch size> \
    --gradient_accumulation_steps <steps> \
    --learning_rate <lr> --weight_decay <wd> --lr_scheduler_type <scheduler> --warmup_ratio <ratio> \
    --max_grad_norm <norm> --cutoff_len <max sequence length> \
    [--do_predict --eval_max_new_tokens <n> --eval_temperature <t> --eval_top_p <p>]
```

The LoRA adapter is saved to `--output_dir`; this directory is the `--stage1_adapter` of Stage 2.

### Stage 2: Iterative Cross-task Learning

> `stage2_cross_task.py` is a re-implementation of Stage 2 written for this release following Section 3.2 of the
> paper. Numbers may differ slightly from those reported in the paper.

**Stage 2 must start from the model trained in Stage 1.** `stage2_cross_task.py` loads the Stage-1 LoRA adapter
(`--stage1_adapter`, required) and continues training it; the LoRA configuration is taken from that adapter.
To run a further round, pass the output directory of a previous Stage-2 run as `--stage1_adapter`.

In every iteration:

1. **TRE batch** (discourse-informed temporal enhancement): the frozen model predicts a connective and discourse
   relation for each event pair; the prediction is appended to the TRE input and the model is fine-tuned to predict
   the gold temporal relation. The parameters are updated.
2. **IDRP batch** (temporal-informed discourse enhancement): the model updated in step 1 extracts cross-argument event
   pairs with their temporal relations in a frozen state; the `(Event A, Event B, Relation)` tuples are appended to
   the IDRP input and the model is fine-tuned to predict the gold connective and discourse relation. The parameters
   are updated.

Auxiliary predictions are validated before they are injected:

- temporal tuples are kept only if both events are concrete words that occur (as whole words) in different
  arguments and the relation is a valid label (placeholders such as `(Event 1, Event 2, before)` are discarded);
- discourse predictions are kept only if they contain a connective and a valid relation label.

Whenever no valid prediction is extracted, the model is guided with increasingly specific questions:

| Level | Guidance |
|---|---|
| 0 | The original auxiliary prompt. |
| 1 | A follow-up question in the same conversation that asks again for concrete events copied from the text (or for a connective and a valid relation) (`--aux_followups`, default 1). |
| 2 | Step-by-step questions. Temporal: the core event of Arg1 → the core event of Arg2 → the relation between the two events. Discourse: the connective → the relation given the connective. Each step is re-asked until its answer is valid (`--aux_step_attempts`, default 3). |
| 3 | Only if all levels fail, the feature is set to `--aux_fallback` (default `None`). |

The guidance level of every test sample is stored in the `guidance` field of `generated_predictions.jsonl`, and the
training log reports how many samples needed each level.

The first-turn prompts follow Figures 4–7 of the paper.

All training and decoding hyper-parameters have to be set by the user; there are no built-in defaults:

```bash
python stage2_cross_task.py \
    --model_name_or_path <base model used in Stage 1> \
    --stage1_adapter <Stage-1 output directory> \
    --pdtb_type pdtb2_l1 --te_task matres \
    --output_dir <output directory> \
    --num_train_epochs <epochs> \
    --per_device_train_batch_size <batch size> \
    --gradient_accumulation_steps <steps> \
    --learning_rate <lr> --weight_decay <wd> --lr_scheduler_type <scheduler> --warmup_ratio <ratio> --max_grad_norm <norm> \
    --cutoff_len <max sequence length> \
    --aux_max_new_tokens <n> --aux_temperature <t> --aux_top_p <p> \
    --eval_max_new_tokens <n> --eval_temperature <t> --eval_top_p <p> \
    --do_predict
```

`--aux_*` controls the frozen auxiliary inference during training and testing, `--eval_*` the final prediction on
the test sets.

With `--do_predict`, the test sets are augmented with auxiliary predictions of the final model and predicted;
the results are written to `<output_dir>/<te_task>_test/` and `<output_dir>/<pdtb_type>_test/` as
`generated_predictions.jsonl` (same format as LLaMA-Factory), which can be scored with `eval.py`.
Run `python stage2_cross_task.py --help` for all options.

## Other files

- `utils.py`: JSON / JSONL I/O and label helpers.
- `wrap.py`: wraps raw data into instruction format with Jinja2 templates.
- `common.py`: code shared by the two training scripts (label sets, Arg1/Arg2 construction, prompts of Figures 2–3, chat formatting, generation).

## Acknowledgements

- The PDTB data processing and evaluation follow [ConnRel](https://github.com/liuwei1206/ConnRel).
- The temporal relation data processing follows [LLMERE](https://github.com/HerbertHu/LLMERE).
- The `eng.pdtb.gum` data and its evaluation follow the [DISRPT 2025 shared task](https://github.com/disrpt/sharedtask2025).

We thank the authors and organisers of these projects for releasing their code and data.

## License

This code is released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{fan-strube-2027-integrating,
  title     = {Integrating Two Linguistic Phenomena: Bi-directional Interplay between Discourse and Temporal Relations},
  author    = {Fan, Yi and Strube, Michael},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```
