import argparse
import json
import os
import re
import string
from collections import defaultdict
from google import genai
from tqdm import tqdm

client = genai.Client(
    vertexai=False,
    api_key=os.environ["GOOGLE_AI_STUDIO"],
)

ALPHABET = [i.upper() for i in string.ascii_lowercase]
FLASH_MODEL = "gemini-2.5-flash-preview-05-20"
PRO_MODEL = "gemini-2.5-pro-preview-05-06"


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
        v = re.findall(r"[\d]+", val)[0]
        if v not in mapper:
            mapper[v] = str(counter + start_from)
            counter += 1
        new_val = re.sub(r"[\d]+", mapper[v], val)

        text = text[: s + offset] + new_val + text[e + offset :]

        offset += len(new_val) - len(val)

    return text, mapper


def sub_pipes(text):
    text = re.sub(r"\|([\d]+)\| ", lambda x: "|" + x.group(1) + "|", text)
    return re.sub(r" \|\|([\d]+)\|\|", lambda x: "|" + x.group(1) + "|", text)


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


def get_first_prompt(novel_data, idx, to_predict):
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

    text = "**Instruction:** You are an excellent linguist working in the field of literature. I will provide you with a passage of a book where some quotes have unique identifiers marked by headers '|quote_id|'. Your are tasked to build a list of quote attributions by sequentially attributing the marked quotes to their speaker.\n\n"
    text += f"**Passage:**\n---\n{context}\n---\n\n"
    text += "**Step 1:** Attribute sequentially each quote to their speaker.\n\n"
    text += "**Step 2:** Match each speaker found in the previous step with one of the following name:\n"
    text += f"---\n{aliases}\n---\n\n"
    text += (
        "**Step 3:** Output a JSON mapping of quote IDs ➔ speakers:\n"
        "{\n"
        "  '1': 'predicted_speaker_1',\n"
        "  '2': 'predicted_speaker_2',\n"
        "}\n\n"
        "Your answer should only contain the output of **Step 3** and can only contain quote identifiers and speakers. Never generate quote content and don't explain your reasoning."
    )

    return text, candidates, is_explicit, truth, inv_mapper


def get_incremental_prompt(novel_data, idx, to_predict, past_preds):
    # 1) Extract & clean the chunk
    context = novel_data["chunks"][idx]
    context = delete_inquote(context)
    context, mapper = restart_quote_seq(context, start_from=1)

    overlap = {
        mapper[q]: speaker for q, speaker in past_preds.items() if q in to_predict
    }
    to_predict = [mapper[q] for q in to_predict]

    context = sub_pipes(context)
    context = clean_context(context)

    candidates = novel_data["candidates"]
    aliases = build_aliases_text(novel_data["aliases"])
    is_explicit = novel_data["is_explicit"]
    inv_mapper = {v: k for k, v in mapper.items()}
    truth = novel_data["speakers_by_chunk"][idx]

    text = "**Instruction:** You are an excellent linguist …\n\n"
    text += f"**Passage:**\n---\n{context}\n---\n\n"
    text += "**Previous predictions:**\n---\n" + repr(overlap) + "\n---\n\n"
    text += (
        "**Step 1:** Attribute each remaining quote to its speaker, "
        "updating any old predictions.\n\n"
    )
    text += "**Step 2:** Match each speaker to one of these names:\n"
    text += f"---\n{aliases}\n---\n\n"
    text += (
        "**Step 3:** Output a JSON mapping of quote IDs ➔ speakers:\n"
        "{\n"
        "  '1': 'predicted_speaker_1',\n"
        "  '2': 'predicted_speaker_2',\n"
        "}\n\n"
        "Your answer should only contain the output of **Step 3** and can only contain quote identifiers and speakers. Never generate quote content and don't explain your reasoning."
    )

    return text, candidates, is_explicit, truth, inv_mapper


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
    all_ids_1 = re.findall(r"[^\|]\|([\d]+)\|[^\|]", text)
    all_ids_2 = re.findall(r"\|\|([\d]+)\|\|", text)

    fq_ids = re.findall(r"\|([\d]+)\| [^\|]+ \|\|[\d]+\|\|", text)

    pb_right = [i for i in all_ids_1 if i not in fq_ids]
    if len(pb_right) > 0:
        for pb in pb_right:
            text = re.sub(rf"[\n]+\|{pb}\|[^\|]+", "", text)
    pb_left = [i for i in all_ids_2 if i not in fq_ids]

    if len(pb_left) > 0:
        for pb in pb_left:
            text = re.sub(rf"[\n]+([^\|]+\|\|{pb}\|\|)", "\n", text)
    return text


def parse_with_reg(preds, mapper):
    p = {}
    preds = re.sub("quote_", "", preds)
    preds = re.sub(r"\|", "", preds)
    for id, char in re.findall(r"\n[\"']?([\d]+)[\"' ]?:[ ]?[^\,]+\n", preds):
        p[mapper[id]] = char
    return p


def parse_with_splits(preds, mapper):
    p = {}
    preds = re.sub("quote_", "", preds)
    preds = re.sub(r"\|", "", preds)
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
        out = re.findall(r"[\n]?([\{]{1}[^\}]+[\}]{1})[\n<]+", out)[0]
        out = re.sub(r"\|", "", out)
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

    model_id = PRO_MODEL if args.use_70b else FLASH_MODEL

    os.makedirs("results", exist_ok=True)

    results = {}
    all_predictions = {}
    novels = data.keys()

    for novel, dataset in data.items():
        past_preds = {}
        all_labels = {}
        explicit_flags = {}
        raw_past_preds = {}

        pbar = tqdm(
            range(len(dataset["chunks"])),
            desc=novel,
            total=len(dataset["chunks"]),
        )

        for idx in pbar:
            to_predict = dataset["to_predict"][idx]
            if not to_predict:
                continue

            overlap = {q: s for q, s in raw_past_preds.items() if q in to_predict}

            if overlap and not args.stop_incremental:
                prompt_text, candidates, is_explicit, labels, id_mapper = (
                    get_incremental_prompt(dataset, idx, to_predict, past_preds=overlap)
                )
            else:
                prompt_text, candidates, is_explicit, labels, id_mapper = (
                    get_first_prompt(dataset, idx, to_predict)
                )

            response = client.models.generate_content(
                model=model_id,
                contents=prompt_text,
            )
            raw_output = response.text
            m = re.search(r"\{.*?\}", raw_output, flags=re.S)
            json_text = m.group(0) if m else "{}"
            parsed = json.loads(json_text)
            raw_predictions = {
                id_mapper[qid.split("_")[-1]]: speaker
                for qid, speaker in parsed.items()
                if qid.isdigit()
            }
            predictions = alias2id(raw_predictions, dataset["name2id"])
            for qid, speaker in predictions.items():
                if qid in to_predict:
                    if args.stop_incremental and (qid in past_preds):
                        continue
                    past_preds[qid] = speaker

            curr_labels = alias2id(labels, dataset["name2id"])
            for qid, speaker in curr_labels.items():
                if (qid in to_predict) and (qid not in all_labels):
                    explicit_flags[qid] = dataset["is_explicit"][qid]
                    all_labels[qid] = speaker

            nexp_preds = {k: v for k, v in past_preds.items() if explicit_flags[k] == 0}
            exp_preds = {k: v for k, v in past_preds.items() if explicit_flags[k] == 1}
            nexp_lab = {k: v for k, v in all_labels.items() if explicit_flags[k] == 0}
            exp_lab = {k: v for k, v in all_labels.items() if explicit_flags[k] == 1}

            try:
                nexp_acc = round(get_acc(nexp_lab, nexp_preds), 3)
            except:
                nexp_acc = "none"
            try:
                exp_acc = round(get_acc(exp_lab, exp_preds), 3)
            except:
                exp_acc = "none"
            try:
                overall_acc = round(get_acc(all_labels, past_preds), 3)
            except:
                overall_acc = "none"

            desc = (
                f"{novel}: #quotes {len(curr_labels)} "
                f"[ChunkAcc] {get_acc(curr_labels, past_preds):0.3f} "
                f"[Acc] {overall_acc} [NexpACC] {nexp_acc} [ExpACC] {exp_acc}"
            )
            pbar.set_description(desc)

        nexp_labels = {k: v for k, v in all_labels.items() if explicit_flags[k] == 0}
        exp_labels = {k: v for k, v in all_labels.items() if explicit_flags[k] == 1}
        nexp_preds2 = {k: v for k, v in past_preds.items() if explicit_flags[k] == 0}
        exp_preds2 = {k: v for k, v in past_preds.items() if explicit_flags[k] == 1}

        results[novel] = {
            "accuracy": get_acc(all_labels, past_preds),
            "nexp_accuracy": get_acc(nexp_labels, nexp_preds2),
            "exp_accuracy": get_acc(exp_labels, exp_preds2),
        }
        all_predictions[novel] = {
            "predictions": past_preds,
            "labels": all_labels,
            "is_explicit": explicit_flags,
        }

        with open(f"results/{args.exp}.res.json", "w") as f:
            json.dump(results, f)
        with open(f"results/{args.exp}.preds.json", "w") as f:
            json.dump(all_predictions, f)
