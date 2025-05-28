import argparse
import json
import os
import re
import string
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_hgface import HGFACE_TOKEN

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
device = torch.device("mps")
ALPHABET = [i.upper() for i in string.ascii_lowercase]
     
torch.manual_seed(42)


def clean_context(text):
    text = re.sub(
        r"([\w\d\.\,\:])[\n]{1}([\w\d\.\,\:])",
        lambda x: x.group(1) + " " + x.group(2),
        text,
    )
    text = re.sub(r"_", " ", text)
    text = re.sub(r"[ ]{3}", " ", text)
    return text


def restart_quote_seq(text, start_from=1):
    mapper = {}
    all_seq = list(re.finditer(r"(\|[\d]+\|)|(\|/[\d]+\|)", text))
    offset = 0
    counter = 0
    for idx, match in enumerate(all_seq):
        s, e = match.span()
        val = match.group(1) if match.group(1) is not None else match.group(2)
        v = re.findall("[\d]+", val)[0]
        if v not in mapper:
            mapper[v] = str(counter + start_from)
            counter += 1
        new_val = re.sub("[\d]+", mapper[v], val)

        text = text[: s + offset] + new_val + text[e + offset :]

        offset += len(new_val) - len(val)

    return text, mapper


def sub_pipes(text):
    text = re.sub("\|([\d]+)\| ", lambda x: "|" + x.group(1) + "|", text)
    return re.sub(" \|\|([\d]+)\|\|", lambda x: "|" + x.group(1) + "|", text)


def alias2id(predictions, name2id):
    processed_preds = {}

    for qid, pred in predictions.items():
        pp = None
        if pred in name2id:
            pp = name2id[pred]
        if pp is None:
            for name, id in name2id.items():
                if pred.lower() in name.lower():
                    pp = id
                    break
        processed_preds[qid] = pp

    return processed_preds


def get_first_prompt(novel_data, idx, tokenizer, to_predict):
    context = novel_data["chunks"][idx]

    context = delete_inquote(context)

    context, mapper = restart_quote_seq(context, start_from=1)
    to_predict = [mapper[i] for i in to_predict]

    context = sub_pipes(context)
    context = clean_context(context)

    candidates = novel_data["candidates"]

    aliases = build_aliases_text(novel_data["aliases"])
    is_explicit = novel_data["is_explicit"]

    inv_mapper = {v: k for k, v in mapper.items()}
    truth = novel_data["speakers_by_chunk"][idx]
    text = """**Instruction:** You are an excellent linguist working in the field of literature. I will provide you with a passage of a book where some quotes have unique identifiers marked by headers '|quote_id|'. Your are tasked to build a list of quote attributions by sequentially attributing the marked quotes to their speaker."""
    text += f"""\n\n**Passage:**
---
{context}
---"""

    text += """\n\n**Step 1:** Attribute sequentially each quote to their speaker."""

    text += """\n\n**Step 2:** Match each speaker found in the previous step with one of the following name:"""

    text += f"\n\n**Names**\n\n---\n{aliases}\n---"

    text += """\n\n**Step 3:** Replace the speakers found in Step 1 with their matching name found in Step 2. Your answer should follow this JSON format:

{
'quote_id_1' : 'predicted_speaker_1',
'quote_id_2' : 'predicted_speaker_2'
}

Your answer should only contain the output of **Step 3** and can only contain quote identifiers and speakers. Never generate quote content and don't explain your reasoning."""

    prompts = [{"role": "user", "content": text}]

    encodings = tokenizer.apply_chat_template(
        prompts, return_tensors="pt", add_generation_prompt=True
    )
    s = encodings.size(1)

    return encodings.to(device), candidates, s, is_explicit, truth, inv_mapper


def get_incremental_prompt(novel_data, idx, tokenizer, to_predict, past_preds):
    context = novel_data["chunks"][idx]

    context = delete_inquote(context)

    context, mapper = restart_quote_seq(context, start_from=1)
    overlap = {mapper[k]: v for k, v in past_preds.items() if k in to_predict}

    to_predict = [mapper[i] for i in to_predict]

    context = sub_pipes(context)
    context = clean_context(context)

    candidates = novel_data["candidates"]

    aliases = build_aliases_text(novel_data["aliases"])
    is_explicit = novel_data["is_explicit"]

    inv_mapper = {v: k for k, v in mapper.items()}

    truth = novel_data["speakers_by_chunk"][idx]

    text = """**Instruction:** You are an excellent linguist working in the field of literature. I will provide you with a passage of a book where some quotes have unique identifiers marked by headers '|quote_id|'. You will also be provided a list of characters and their aliases, and previous predictions. Your are tasked to build a list of quote attributions by sequentially attributing the marked quotes to their speaker."""

    text += f"""\n\n**Passage:**
---
{context}
---"""

    text += f"\n\n**Previous predictions**\n\n---\n{overlap}\n---"

    text += """\n\n**Step 1:** Attribute sequentially each quote to their speaker. Update the previous predictions if you think it contains wrong speaker prediction."""

    text += """\n\n**Step 2:** Match each speaker found in the previous step with one of the following name:"""

    text += f"\n\n**Names**\n\n---\n{aliases}\n---"

    text += """\n\n**Step 3:** Replace the speakers found in Step 1 with their matching name found in Step 2. Your answer should follow this JSON format:

{
'quote_id_1' : 'predicted_speaker_1',
'quote_id_2' : 'predicted_speaker_2'
}

Your answer should only contain the output of **Step 3** and can only contain quote identifiers and speakers. Never generate quote content and don't explain your reasoning."""

    prompts = [{"role": "user", "content": text}]

    encodings = tokenizer.apply_chat_template(
        prompts, return_tensors="pt", add_generation_prompt=True
    )
    s = encodings.size(1)

    return encodings.to(device), candidates, s, is_explicit, truth, inv_mapper


def get_acc(labels, preds):
    correct = 0
    total = 0
    for idx in labels.keys():
        if idx in preds.keys():
            if preds[idx] == labels[idx]:
                correct += 1
        total += 1
    return correct / total


def build_aliases_text(aliases):
    rev_dic = defaultdict(list)
    for k, v in aliases.items():
        if k not in ["_narr", "_group", "_unknowable"]:
            rev_dic[v].append(k)
    text = ""
    length = len(rev_dic)

    for idx, (speaker, al) in enumerate(rev_dic.items()):
        text += f"{speaker}"
        if len([k for k in al if k != speaker]) > 0:
            text += "=" + "=".join([k for k in al if k != speaker])
        if idx < (length - 1):
            text += "\n"
    return text + ""


def delete_inquote(text):
    all_ids_1 = re.findall("[^\|]\|([\d]+)\|[^\|]", text)
    all_ids_2 = re.findall("\|\|([\d]+)\|\|", text)

    fq_ids = re.findall("\|([\d]+)\| [^\|]+ \|\|[\d]+\|\|", text)

    pb_right = [i for i in all_ids_1 if i not in fq_ids]
    if len(pb_right) > 0:
        for pb in pb_right:
            text = re.sub(f"[\n]+\|{pb}\|[^\|]+", "", text)
    pb_left = [i for i in all_ids_2 if i not in fq_ids]

    if len(pb_left) > 0:
        for pb in pb_left:
            text = re.sub(f"[\n]+([^\|]+\|\|{pb}\|\|)", "\n", text)
    return text


def parse_with_reg(preds, mapper):
    p = {}
    preds = re.sub("quote_", "", preds)
    preds = re.sub("\|", "", preds)
    for id, char in re.findall("\n[\"']?([\d]+)[\"' ]?:[ ]?[^\,]+\n", preds):
        p[mapper[id]] = char
    return p


def parse_with_splits(preds, mapper):
    p = {}
    preds = re.sub("quote_", "", preds)
    preds = re.sub("\|", "", preds)
    for splitted in preds.split("\n"):
        if ":" in splitted:
            qid = splitted.split(":")[0].strip()
            if any(['"' in qid[0], "'" in qid[0]]):
                qid = eval(qid)
            try:
                q = int(qid)
            except:
                continue

            pred = splitted.split(":")[1].strip()
            if len(pred) > 0:
                if pred[-1] == ",":
                    pred = pred[:-1]
                if any(['"' in pred[0], "'" in pred[0]]):
                    pred = pred[1:-1]
            p[mapper[qid]] = pred
    return p


def parse_answer(out, mapper):
    try:
        print(out)

        out = re.findall("[\n]?([\{]{1}[^\}]+[\}]{1})[\n<]+", out)[0]
        out = re.sub("\|", "", out)
        out = re.sub("quote_", "", out)
        out = eval(out)
        preds = {}
        for qid, speaker in out.items():
            qid = str(qid)
            if qid in mapper:
                preds[mapper[qid]] = speaker
    except:
        try:
            preds = parse_with_splits(out, mapper)
        except:
            try:
                preds = parse_with_reg(out, mapper)
            except:
                preds = {}

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        help="path to data stored in json",
        default="../data/seen.all.strided.1024.pdnc.4096.json",
        type=str,
    )
    parser.add_argument("--exp", help="experiment name", type=str, required=True)
    parser.add_argument(
        "--stop_incremental", help="do not use incremental prompt", action="store_true"
    )
    parser.add_argument(
        "--use_70b",
        help="do you want to use 70b model?",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    split = 0
    with open(args.data_path, "r") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=HGFACE_TOKEN, torch_dtype=torch.float16
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not os.path.exists("results/"):
        os.makedirs("results/")

    if args.use_70b:
        model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        is_decoder=True,
        token=HGFACE_TOKEN,
        device_map={"": device},
        torch_dtype=torch.float16,
    ).eval()

    results = {}
    all_predictions = {}
    novels = data.keys()

    for novel in list(novels):
        dataset = data[novel]
        past_preds = {}
        all_labels = {}
        explicit = {}
        raw_past_preds = {}
        pbar = tqdm(
            enumerate(range(len(dataset["chunks"]))),
            desc=f"{novel}",
            total=len(dataset["chunks"]),
        )

        cnt = 0
        for c, idx in pbar:
            if len(dataset["to_predict"][idx]) > 0:
                overlap = {
                    k: v
                    for k, v in raw_past_preds.items()
                    if k in dataset["to_predict"][idx]
                }
                if len(overlap) > 0:
                    if args.stop_incremental:
                        prompts, candidates, s, ie, labels, id_mapper = (
                            get_first_prompt(
                                dataset, idx, tokenizer, dataset["to_predict"][idx]
                            )
                        )
                    else:
                        prompts, candidates, s, ie, labels, id_mapper = (
                            get_incremental_prompt(
                                dataset,
                                idx,
                                tokenizer,
                                dataset["to_predict"][idx],
                                past_preds=overlap,
                            )
                        )
                else:
                    prompts, candidates, s, ie, labels, id_mapper = get_first_prompt(
                        dataset, idx, tokenizer, dataset["to_predict"][idx]
                    )
                if c < 2:
                    print(tokenizer.batch_decode(prompts)[0])
                with torch.no_grad():
                    out = model.generate(prompts, max_new_tokens=100, do_sample=False)
                predictions = tokenizer.batch_decode(out[:, s:])[0]
                print(predictions)

                try:
                    raw_predictions = parse_answer(predictions, id_mapper)
                    print(raw_predictions)
                    raw_past_preds.update(raw_predictions)
                    predictions = alias2id(raw_predictions, data[novel]["name2id"])
                    print(predictions)
                except:
                    predictions = {}

                for i in predictions.keys():
                    if i in dataset["to_predict"][idx]:
                        if (args.stop_incremental) & (i in past_preds):
                            continue
                        else:
                            past_preds[i] = predictions[i]

                curr_lab = alias2id(labels, data[novel]["name2id"])
                for i in curr_lab.keys():
                    if (i in dataset["to_predict"][idx]) & (i not in all_labels):
                        explicit[i] = dataset["is_explicit"][i]
                        all_labels[i] = curr_lab[i]

                nexp_preds = {k: v for k, v in past_preds.items() if explicit[k] == 0}
                exp_preds = {k: v for k, v in past_preds.items() if explicit[k] == 1}

                nexp_lab = {k: v for k, v in all_labels.items() if explicit[k] == 0}
                exp_lab = {k: v for k, v in all_labels.items() if explicit[k] == 1}

                try:
                    nexp_acc = round(get_acc(nexp_lab, nexp_preds), 3)
                except:
                    nexp_acc = "none"
                try:
                    exp_acc = round(get_acc(exp_lab, exp_preds), 3)
                except:
                    exp_acc = "none"
                try:
                    acc = round(get_acc(all_labels, past_preds), 3)
                except:
                    acc = "none"
                pbar.set_description(
                    f"{novel}: # quotes {len(curr_lab)} [ChunkAcc] {get_acc(curr_lab, past_preds):0.3f} [Acc] {acc} [NexpACC] {nexp_acc} [ExpACC] {exp_acc}"
                )

        exp_labels = {k: v for k, v in all_labels.items() if explicit[k] == 1}
        nexp_labels = {k: v for k, v in all_labels.items() if explicit[k] == 0}

        exp_preds = {k: v for k, v in past_preds.items() if explicit[k] == 1}
        nexp_preds = {k: v for k, v in past_preds.items() if explicit[k] == 0}

        nexp_accuracy = get_acc(nexp_labels, nexp_preds)
        exp_accuracy = get_acc(exp_labels, exp_preds)
        overall_acc = get_acc(all_labels, past_preds)

        print(
            f"\n{novel}: [ACC: {overall_acc:0.3f}] [NEXPACC: {nexp_accuracy:0.3f}] [EXPACC: {exp_accuracy:0.3f}]"
        )

        results[novel] = {
            "accuracy": overall_acc,
            "exp_accuracy": exp_accuracy,
            "nexp_accuracy": nexp_accuracy,
        }
        all_predictions[novel] = {
            "predictions": past_preds,
            "labels": all_labels,
            "is_explicit": explicit,
        }

        with open(f"results/{args.exp}.res.json", "w") as f:
            json.dump(results, f)
        with open(f"results/{args.exp}.preds.json", "w") as f:
            json.dump(all_predictions, f)
