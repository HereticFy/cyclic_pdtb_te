"""External paths used by this project.

Set each path either via the environment variable of the same name, e.g.

    export LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory

or by editing the default values below.
"""
import os

# Root of your LLaMA-Factory checkout. Generated training data is written here,
# and model predictions are read from its `saves/` directory.
LLAMA_FACTORY_DIR = os.environ.get("LLAMA_FACTORY_DIR", "path/to/LLaMA-Factory")

# Directory containing the raw PDTB data, organised as <task_name>/<split>.json
# (e.g. pdtb2/train.json, pdtb3/test.json).
PDTB_SOURCE_DIR = os.environ.get("PDTB_SOURCE_DIR", "path/to/pdtb_source_data")

# DISRPT shared task data directory, organised as <dataset>/<dataset>_<split>.{conllu,tok,rels}.
DISRPT_DIR = os.environ.get("DISRPT_DIR", "path/to/disrpt/data")

# DISRPT data directory used for the eng.pdtb.gum test files.
DISRPT_TEST_DIR = os.environ.get("DISRPT_TEST_DIR", "path/to/disrpt_test/data")
