import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import string
import argparse
from functools import partial
from llama3_utils import parse_model_answer, PIPELINE, get_explicit
import os
from token_hgface import HGFACE_TOKEN
import re
from collections import defaultdict

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = torch.device("cuda:0")
ALPHABET = [i.upper() for i in string.ascii_lowercase]


torch.manual_seed(42)


def sub_explicits(text, remove=False):
    # return re.sub("said by ([\w\s\.]+)\|\|", "", text)
    if remove:
        return re.sub("\| said by ([^\|]+)\s\|", "|", text)
        # return re.sub("\|\s([^\|]+)\s\|", "", text)
    else:
        return re.sub(
            "\| said by ([^\|]+)\s\|", lambda x: "[" + x.group(1).upper() + "] |", text
        )
    # return re.sub("\|\s([^\|]+)\s\|", lambda x: " " + x.group(1) +" ||", text)


def remove_exp(text):
    return re.sub("\| said by ([^\|]+)\s\|", "|", text)


def add_preds_in_context(text, past_preds):
    quotes_in_ctx = set(re.findall(r"\|([\d]+)\| ", text))
    for qid, pred in past_preds.items():
        qqid = f"|{qid}| "
        qqid2 = f" ||{qid}||"

        if qid in quotes_in_ctx:
            text = text.replace(qqid, f"|{qid}| <{pred.upper()}> ")
            text = text.replace(qqid2, f" <{pred.upper()}> |{qid}|")

    return text


def sub_explicits(text, explicit_quotes):
    quotes_in_ctx = set(re.findall(r"\|([\d]+)\| ", text))
    for qid, speaker in explicit_quotes.items():
        qqid = f"|{qid}| "
        qqid2 = f" ||{qid}||"

        if qid in quotes_in_ctx:
            text = text.replace(qqid, f"|{qid}| <{speaker.upper()}> ")
            text = text.replace(qqid2, f" <{speaker.upper()}> |{qid}|")
    return text


def sub_to_attribute(text, to_predict):
    quotes_in_ctx = set(re.findall(r"\|([\d]+)\| ", text))
    for qid in to_predict:
        qqid = f"|{qid}| "
        qqid2 = f" ||{qid}||"

        if qid in quotes_in_ctx:
            text = text.replace(qqid, f"|{qid}| <to attribute>  ")
            # text = text.replace(qqid2, f" </TO ATTRIBUTE>  |/{qid}|")
            # text = text.replace(qqid2, f" |{qid}| [{pred.upper()}]")
    return text


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
    spans = []
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


def lenient_replace_aliases(predictions, gold_aliases):

    aliases = gold_aliases.copy()
    for k, v in gold_aliases.items():
        aliases[k.lower()] = v

    processed_preds = {}

    for qid, pred in predictions.items():
        if pred.lower() in aliases:
            processed_preds[qid] = aliases[pred.lower()]
        else:
            processed_preds[qid] = pred

    return processed_preds


def alias2id(predictions, name2id):

    # aliases = name2id.copy()
    # for k, v in name2id.items():
    #     aliases[k.lower()] = v

    processed_preds = {}

    # First try exact match
    for qid, pred in predictions.items():
        # pp = None
        pp = None
        if pred in name2id:
            pp = name2id[pred]
        # If no match, try lenient metric
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

    text += f"\n\n**Step 1:** Attribute sequentially each quote to their speaker."
    
    text += f"\n\n**Step 2:** Match each speaker found in the previous step with one of the following name:"
    
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
    overlap = {mapper[k]:v for k,v in past_preds.items() if k in to_predict}

    to_predict = [mapper[i] for i in to_predict]

    context = sub_pipes(context)
    context = clean_context(context)

    candidates = novel_data["candidates"] 
    
    aliases = build_aliases_text(novel_data["aliases"])
    is_explicit = novel_data["is_explicit"]

    
    inv_mapper = {v: k for k, v in mapper.items()}

    truth = novel_data["speakers_by_chunk"][idx]

    text = """**Instruction:** You are an excellent linguist working in the field of literature. I will provide you with a passage of a book where some quotes have unique identifiers marked by headers '|quote_id|'. You will also be provided a list of characters and their aliases, and previous predictions. Your are tasked to build a list of quote attributions by sequentially attributing the marked quotes to their speaker."""
    
    # text += f"\n\n**Candidate characters and their aliases:**\n\n---\n{aliases}\n---"
    
    text += f"""\n\n**Passage:**
---
{context}
---"""

#     text += f"\n\n**Step 1:** Process quote-by-quote and chose a speaker for each quote from the list of candidate of characters above. You can update the previous predictions if you think it contains wrong speaker prediction."
    
#     text += """\n\n**Step 2:** Structure your answer as the following JSON format:

# {
# 'quote_id_1' : 'predicted_speaker_1',
# 'quote_id_2' : 'predicted_speaker_2'
# }

# Your answer should only contain the output of **Step 2** and can only contain quote identifiers and speakers from the list of characters. Never generate quote content and don't explain your reasoning."""

    text += f"\n\n**Previous predictions**\n\n---\n{overlap}\n---"

    text += f"\n\n**Step 1:** Attribute sequentially each quote to their speaker. Update the previous predictions if you think it contains wrong speaker prediction."
    
    text += f"\n\n**Step 2:** Match each speaker found in the previous step with one of the following name:"
    
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
    # rev_aliases = defaultdict(list)

    # for k,v in aliases.items() :
    #     rev_aliases[v].append(k.replace("_narr", "The narrator"))
    # text = ""
    # for k,v in rev_aliases.items() :
    #     vv = ", ".join(v)
    #     text += f"{k}: {vv}\n"
    # text = "alias = canonical speaker name\n"
    rev_dic = defaultdict(list)
    for k, v in aliases.items():
        if k not in ["_narr", "_group", "_unknowable"]:
            rev_dic[v].append(k)
    # aliases = rev_dic.values()

    # else :
    #     rev_dic[v].append(k)
    # text += f'"{v}" - "{k.replace("_narr", "The narrator")}"\n'
    # text += f'{k.replace("_narr", "The narrator")} = {v}\n'
    # text = "Speaker NameAliases\n\n"
    text = ""
    length = len(rev_dic)
    test = []

    for idx, (speaker, al) in enumerate(rev_dic.items()):
        # for aa in al:
        #     test.append(aa)
        # j = str(set([speaker] + al))
        # text += f"'{speaker}' :" + ", ".join(al)
        # al = ", ".join([ '"' + i + '"' for i in al])
        text += f"{speaker}"
        if len([k for k in al if k!= speaker]) > 0 : 
            text += "=" + "=".join([k for k in al if k!= speaker])
        if idx < (length - 1):
            text += "\n"
        # text += f"{speaker}: {al}\n"
    return text + ""


def old_build_aliases_text(aliases):
    # rev_aliases = defaultdict(list)

    # for k,v in aliases.items() :
    #     rev_aliases[v].append(k.replace("_narr", "The narrator"))
    # text = ""
    # for k,v in rev_aliases.items() :
    #     vv = ", ".join(v)
    #     text += f"{k}: {vv}\n"
    # text = "alias = canonical speaker name\n"
    rev_dic = defaultdict(list)
    for k, v in aliases.items():
        if k not in ["_narr", "_group", "_unknowable"]:
            rev_dic[v].append(k)
        # else :
        #     rev_dic[v].append(k)
        # text += f'"{v}" - "{k.replace("_narr", "The narrator")}"\n'
        # text += f'{k.replace("_narr", "The narrator")} = {v}\n'
    text = "Speaker Name: Aliases\n\n"
    for speaker, al in rev_dic.items():
        # al = ", ".join([ '"' + i + '"' for i in al])
        al = ", ".join(al)

        text += f"{speaker}: {al}\n"
    return text


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


def remove_unpredictable(context, to_predict):
    all_ids_1 = re.findall("[^\|]\|([\d]+)\|[^\|]", context)
    all_ids_2 = re.findall("\|\|([\d]+)\|\|", context)
    all_ids = list(set(all_ids_1) & set(all_ids_2))

    for id in [i for i in all_ids if i not in to_predict]:
        context = re.sub(f"\|{id}\| ", "", context)
        context = re.sub(f" \|\|{id}\|\|", "", context)
    return context


def parse_with_reg(preds, mapper):
    p = {}
    preds = re.sub("quote_", "", preds)
    preds = re.sub("\|", "", preds)
    for id, char in re.findall(
        "\n[\"\']?([\d]+)[\"\' ]?:[ ]?[^\,]+\n", preds
    ):
        p[mapper[id]] = char
    return p


def parse_with_splits(preds, mapper) : 
    
    p = {}
    preds = re.sub("quote_", "", preds)
    preds = re.sub("\|", "", preds)
    for splitted in preds.split("\n") : 
        if ":" in splitted : 
            qid = splitted.split(":")[0].strip()
            if any(["\"" in qid[0], "\'" in qid[0]]) :
                qid = eval(qid)
            try : 
                q = int(qid)
            except : 
                continue
            
            pred = splitted.split(":")[1].strip()
            if len(pred) > 0 : 

                if pred[-1] == "," : 

                    pred = pred[:-1]
                if any(["\"" in pred[0], "\'" in pred[0]]) : 
                    pred = pred[1:-1]
                # pred = eval(pred)
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
            try : 

                preds = parse_with_reg(out, mapper)
            except : 
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
    parser.add_argument("--stop_incremental", help="do not use incremental prompt", action="store_true")
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
        model_name,
        token=HGFACE_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not os.path.exists("results/"):
        os.makedirs("results/")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    if args.use_70b:
        model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        is_decoder=True,
        token=HGFACE_TOKEN,
        device_map="cuda:0",
        quantization_config=quantization_config,
    ).eval()

    results = {}
    all_predictions = {}
    # novels = set(dataset.novel_id)
    # all_indices = {i:get_indices(dataset,i) for i in novels}
    novels = data.keys()
    # novels = [i for i in novels if i not in ["ARoomWithAView", "AgeOfInnocence", "AliceInWonderland"]]
    
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

        # pbar = tqdm(enumerate(range(13, len(dataset["chunks"]))), desc=f"{novel}", total=len(dataset["chunks"]))
        cnt = 0 
        for c, idx in pbar:
            # if c == 21 :
            #     breakW
            if len(dataset["to_predict"][idx]) > 0:
                overlap = {k:v for k,v in raw_past_preds.items() if k in dataset["to_predict"][idx]}
                if len(overlap) > 0 : 
                    if args.stop_incremental : 
                        prompts, candidates, s, ie, labels, id_mapper = get_first_prompt(
                            dataset, idx, tokenizer, dataset["to_predict"][idx]
                        )
                    else : 
                        prompts, candidates, s, ie, labels, id_mapper = get_incremental_prompt(
                            dataset, idx, tokenizer, dataset["to_predict"][idx], past_preds=overlap
                        )
                else : 
                    prompts, candidates, s, ie, labels, id_mapper = get_first_prompt(
                        dataset, idx, tokenizer, dataset["to_predict"][idx]
                    )
                if c < 2:
                    print(tokenizer.batch_decode(prompts)[0])
                # Do not process chunks with no quotes to be predicted
                with torch.no_grad():
                    out = model.generate(prompts, max_new_tokens=2000, do_sample=False)
                predictions = tokenizer.batch_decode(out[:, s:])[0]
                print(predictions)

                # predictions = parse_model_answer(predictions, id_mapper, pipeline=PIPELINE, debug=False)
                try:
                    raw_predictions = parse_answer(predictions, id_mapper)
                    print(raw_predictions)
                    raw_past_preds.update(raw_predictions)
                    predictions = alias2id(raw_predictions, data[novel]["name2id"])
                    print(predictions)
                except:
                    predictions = {}
                # print(predictions)
                
                
                for i in predictions.keys():

                    # Only add predictions
                    if (i in dataset["to_predict"][idx]):
                        if (args.stop_incremental) & (i in past_preds) : 
                            continue
                        else : 
                            past_preds[i] = predictions[i]

                curr_lab = alias2id(labels, data[novel]["name2id"])
                # print(curr_lab)
                for i in curr_lab.keys():
                    # if dataset["is_explicit"][i] == 0 :
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
        #             print(f"Accuracy : {correct / total :0.3f}")
        #             print(f"NEXP Accuracy : {correct_nexp / total_nexp :0.3f}")

        # nexp_labels = {
        #     idx: dataset["speakers"][idx].lower()
        #     for idx in dataset["speakers"]
        #     if dataset["is_explicit"][idx] == 0
        # }
        # nexp_labels = alias2id(nexp_labels,  data[novel]["name2id"])
        # nexp_preds = {
        #     k: v for k, v in past_preds.items() if dataset["is_explicit"][k] == 0
        # }

        # exp_labels = {
        #     idx: dataset["speakers"][idx].lower()
        #     for idx in dataset["speakers"]
        #     if dataset["is_explicit"][idx] == 1
        # }
        exp_labels = {k: v for k, v in all_labels.items() if explicit[k] == 1}
        nexp_labels = {k: v for k, v in all_labels.items() if explicit[k] == 0}

        # exp_labels = alias2id(exp_labels,  data[novel]["name2id"])

        exp_preds = {k: v for k, v in past_preds.items() if explicit[k] == 1}
        nexp_preds = {k: v for k, v in past_preds.items() if explicit[k] == 0}

        nexp_accuracy = get_acc(nexp_labels, nexp_preds)
        exp_accuracy = get_acc(exp_labels, exp_preds)
        overall_acc = get_acc(all_labels, past_preds)

        print(
            f"\n{novel}: [ACC: {overall_acc :0.3f}] [NEXPACC: {nexp_accuracy :0.3f}] [EXPACC: {exp_accuracy :0.3f}]"
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

        # print(f"{novel}: NEXP Accuracy : {n_correct_nexp / n_total_nexp :0.3f}")
        # print(f"[OVERALL] NEXP Accuracy : {correct_nexp / total_nexp :0.3f}")
