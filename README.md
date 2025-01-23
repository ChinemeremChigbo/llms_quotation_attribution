This is the official repository for the paper [*Evaluating LLMs for Quotation Attribution in Literary Texts: A Case Study of LLaMa3*](https://arxiv.org/abs/2406.11380), G. Michel, E. V. Epure, R. Hennequin and C. Cerisara (NAACL 2025).

In this paper, we present experiments where we prompt Llama-3 8b and 70b in a zero-shot setting to attribute quotes in literary texts. We then evaluate the impact of memorized information (book memorization & data contamination) on the downstream performance.

This repository contains scripts to reproduce our main experiments.

# Prompts

You can find the prompts used in our experiments in the folder `prompts/`.

# Installation

Run the following commands to create an environment and install all the required packages:

```bash
python3 -m venv llms_for_qa
. ./llms_for_qa/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
```

# Data 

We provide in the `/data` folder all data used for our experiments.
In particular, the annotations and full text of the recently published novel, Dark Corners, that we annotated manually are available in the folder `unseen_source/`. We followed the PDNC guidelines, but did not annotated mentions of entities within quotes.

The folder `pdnc_source` contains data taken from [this repository](https://github.com/Priya22/speaker-attribution-acl2023/tree/main/data/pdnc_source).
We aligned the annotations from the [official PDNC github repository](https://github.com/Priya22/project-dialogism-novel-corpus/tree/master) to match the annotations in `pdnc_source` for the 6 novels of the second release of PDNC and Dark Corners in `test_pdnc_source`.

The file `novel_metadata.json` contain book titles and author for each novel in our dataset. 

In each novel folder, we provide the raw text, quotation annotations, character list as well as the passages used for testing name-cloze (in `$novel/$novel.name_cloze.txt`)

We also ran BookNLP on each novel, giving two additional files: `novel.entities` containing results of NER and coreference resolution and `novel.tokens` containing each processed tokens by BookNLP. 

For our Corrupted-Speaker-Guessing experiments, we manually annotated name replacements that can be found in the file `id2replacements.pkl`.

The files *seen.all.strided.1024.pdnc.4096.right0.json*, *test.all.strided.1024.pdnc.4096.right0.json* and *unseen.all.strided.1024.pdnc.4096.right0.json* contains the processed chunks for PDNC_1, PDNC_2 and Dark Corners (the Unseen Novel) respectively.

# Scripts

We provide bash scripts to reproduce each of our experiments: 

`attribution.sh` for Section 4.

`memorization.sh` for Section 5 - Book Memorization.

`contamination.sh` for Section 5 - Annotation Contamination.

Before running an experiment, [make sur you have access to Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B), and insert your hugginface token by modifying the value of `HGFACE_TOKEN` in the file `token_hgface.py`.
