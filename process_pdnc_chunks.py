import tqdm
import string
import pickle as pkl
import pandas as pd
import os
import string
import json
from collections import defaultdict
import argparse
from transformers import AutoTokenizer
import re
from token_hgface import HGFACE_TOKEN


def get_offset_bytes(qtext, sb, eb):
    i = 0
    while qtext[i] in string.whitespace:
        i += 1
    sb = sb + i

    i = 0
    while qtext[-(i + 1)] in string.whitespace:
        i += 1
    eb = eb - i
    return sb, eb


def get_enhanced_char_list(charInfo, add_lowercase=False):
    """This function processes the original character list to extend it to additional cases. Examples:

"_narr" --> ["Narr", "Narrator", "I"]
"Miss. Anne Eliot" --> ["Anne Eliot"]"""
    PREFIXES = ['Mr.', 'Mrs.', 'Miss.', 'Lady', 'Sir', 'Mrs', 'Mr', 'Miss', 'Dr.', 'Dr', 'Madame', 'Madam', \
            'Mademoiselle', 'St.', 'St', 'Ms.', 'Ms', 'Count', 'Countess', "The"]
    PREFIXES.extend([x.lower() for x in PREFIXES])

    #don't include lowercase because names can also be common nouns sometimes (Lily, Rose)
    enhanced_name2id = {}
    name2id = charInfo["name2id"]
    new_cands = {}
    
    for name, id_ in name2id.items():
        if name in ["_narr", "Narr"] : 
            enhanced_name2id["Narrator"] = id_
            enhanced_name2id["The Narrator"] = id_
            enhanced_name2id["I"] = id_
            narr_id = id_
        else : 
            enhanced_name2id[name] = id_
            
        if add_lowercase:
            enhanced_name2id[name.lower()] = id_
    if any(["_narr" in charInfo["id2parent"].values(), "Narr" in charInfo["id2parent"].values()]) :
        charInfo["id2parent"][narr_id] = "Narrator"
    
    for name, id_ in name2id.items():
        
        n_words = name.split()
        if n_words[0] in PREFIXES:
            new_cand = " ".join(n_words[1:])

            if (len(new_cand)>0) and (new_cand not in enhanced_name2id):
                if new_cand not in new_cands:
                    new_cands[new_cand] = []
                new_cands[new_cand].append(id_)

                if add_lowercase:
                    if new_cand.lower() not in new_cands:
                        new_cands[new_cand.lower()] = []

                    new_cands[new_cand.lower()].append(id_)
        
    
    for new_cand, ids in new_cands.items():
        ids = list(set(ids))
        if len(ids) == 1:
            enhanced_name2id[new_cand] = ids[0]

    print("original count: {} ; enhanced count: {}".format(len(name2id), len(enhanced_name2id)))
    
    enhanced_id2names = {}
    for n, i in enhanced_name2id.items():
        if i not in enhanced_id2names:
            enhanced_id2names[i] = set()
        enhanced_id2names[i].add(n)
    
    return {'name2id': enhanced_name2id, 'id2names': enhanced_id2names, "id2parent" : charInfo["id2parent"], "id2gender" : charInfo["id2gender"]}

def iterate(encoding, char, how="left"):
    """Iterate over a transformers.tokenizer.Encoding to returns the token at position `char` or its nearest (left or right) token."""
    max_length = len(encoding.ids) - 1
    max_char_length = max(encoding.offsets)[0]
    input = None
    for it in range(1, max_length):
        if input is not None:
            return input

        if how == "left":
            input = encoding.char_to_token(char - it)
        else:
            input = encoding.char_to_token(char + it)

        if all([how == "left", char - it <= 0]):
            input = 0
        elif all([how == "right", char + it >= max_char_length]):
            input = max_length

    if not input:
        if how == "left":
            return 0
        else:
            return max_length


def read_quotes(root_folder, novel):
    """Reads the data stored in PDNC"""
    quoteinf = []

    # for novel in NOVELS:
    quotedf = pd.read_csv(os.path.join(root_folder, novel, "quote_info.csv"))
    charInf = pkl.load(
        open(os.path.join(root_folder, novel, "charInfo.dict.pkl"), "rb")
    )

    for _, row in quotedf.iterrows():
        sb, eb = eval(row["qSpan"])
        qtt = row["qText"]
        asb, aeb = get_offset_bytes(qtt, sb, eb)
        qType = row["qType"]
        qId = row["qID"]
        qChapId = row["startChapID"]
        # try:
        # st, et = mappers[asb], mappers[aeb-1]
        speaker = row["speaker"]
        speaker_id = charInf["name2id"][speaker]
        eid = "CHAR_" + str(speaker_id)
        etype = row["speakerType"]
        adressees = eval(row["addressee"])
        canonical_speaker_name = charInf["id2parent"][speaker_id]
        quoteinf.append(
            (
                qId,
                qChapId,
                asb,
                aeb,
                eid,
                qType,
                speaker,
                adressees,
                etype,
                canonical_speaker_name,
            )
        )
        # Remove minor characters if asked
        # if use_minor_char:
        # 	quoteinf.append((qId, qChapId, st, et, eid, qType, speaker, adressees))
        # elif etype != "minor":
        # 	quoteinf.append((qId, qChapId, st, et, eid, qType, speaker, adressees))

        # except KeyError:
        #     print("Error!!! Novel {} qID {}".format(novel, qId))

    print("Read {}/{} quotes".format(len(quoteinf), len(quotedf)))
    return quoteinf


def prefix_suffix_quote_id(text, q_char_s, q_char_e, counter, speaker=None):
    """Replace the original text by adding prefixes and suffixes to the quote starting at position `q_char_s` and `q_char_e`. Also add another prefix if provided a `speaker` name. 
The replacement will look like that: '<text> [q] <text>' --> '<text> |`counter`| [q] ||`counter`|| <text>' where [q] is the target quote."""
    if speaker is not None:
        prompt = f"| said by {speaker} |{counter}| "
        end_prompt = f" ||{counter}||"

    else:
        prompt = f"|{counter}| "
        end_prompt = f" ||{counter}||"

    if (text[q_char_e] == '"') & (text[q_char_s - 1] == '"'):
        j = (
            text[: q_char_s - 1]
            + prompt
            + text[q_char_s - 1 : q_char_e + 1]
            + end_prompt
            + text[q_char_e + 1 :]
        )
        length = len(prompt) + len(end_prompt)
    elif (text[q_char_e] != '"') & (text[q_char_s - 1] != '"'):
        j = (
            text[:q_char_s]
            + " "
            + prompt
            + text[q_char_s:q_char_e]
            + end_prompt
            + " "
            + text[q_char_e:]
        )
        # print("2", prompt + text[q_char_s:q_char_e] + end_prompt)
        length = len(prompt) + len(end_prompt) + 2
    elif text[q_char_s - 1] != '"':
        j = (
            text[:q_char_s]
            + prompt
            + text[q_char_s : q_char_e + 1]
            + end_prompt
            + text[q_char_e + 1 :]
        )
        length = len(prompt) + len(end_prompt)
    elif text[q_char_e] != '"':
        j = (
            text[: q_char_s - 1]
            + prompt
            + text[q_char_s - 1 : q_char_e]
            + end_prompt
            + text[q_char_e:]
        )
        length = len(prompt) + len(end_prompt)

    return j, length


def prune_info(speaker_info, speakers) : 
    """Given a dictionary containing the character aliases, returns a subset of this dictionary for the provided `speakers`."""
    valid_ids = [speaker_info["name2id"][s] for s in speakers]

    id2parent = {k:v for k,v in speaker_info["id2parent"].items() if k in valid_ids}
    id2names = {k:v for k,v in speaker_info["id2names"].items() if k in valid_ids}
    name2id = {k:v for k,v in speaker_info["name2id"].items() if v in valid_ids}
    id2gender = {k:v for k,v in speaker_info["id2gender"].items() if k in valid_ids}

    print(f"original {len(speaker_info['name2id'])} ; pruned {len(speaker_info['name2id']) - len(name2id)}")

    return {"id2parent" : id2parent, "id2names" : id2names, "name2id":name2id, "id2gender" : id2gender} 
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data", help="Filename containing source data")
    parser.add_argument("--save_path", help="path to save json file")
    parser.add_argument(
        "--max_length", help="Max length of context + quote", type=int, default=4096
    )
    parser.add_argument(
        "--strides", help="Length of stride", type=int, default=1024
    )
    parser.add_argument(
        "--right_context_length",
        help="Max length of quote + right_context",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--model_name",
        help="model huggingface name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--avoid_long_tail",
        help="avoid minor characters",
        default=True,
        action="store_false",
    )
    parser.add_argument(
        "--only_nexp",
        help="whether to only consider non-explicit quotes.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--predict_all",
        help="whether to consider all quotes to be predicted, even if striding.",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, token=HGFACE_TOKEN
    )

    data = defaultdict(list)
    max_context = args.max_length
    novels = [
        i
        for i in os.listdir(args.source_data)
        if os.path.isdir(os.path.join(args.source_data, i))
    ]
    
    for novel in tqdm.tqdm(novels):
        print(novel.upper())
        with open(os.path.join(args.source_data, novel, "novel.txt")) as f:
            text = f.read()
        quotes = read_quotes(args.source_data, novel)
        
        # we also remove "_narr" which means that the narrator is never given a name. 
        valid_quotes = [
            q for q in quotes if q[6] not in ["_group", "_unknowable", "_narr"]
        ]
        speakers = set([q[6] for q in valid_quotes])
        
        with open(
            os.path.join(args.source_data, novel, "charInfo.dict.pkl"), "rb"
        ) as f:
            speaker_info = pkl.load(f)
        
        speaker_info = prune_info(speaker_info, speakers)
        speaker_info = get_enhanced_char_list(speaker_info)
        
        chap_info = pd.read_csv(os.path.join(args.source_data, novel, "chap_info.csv"))

        ends = []
        starts = []
        counter = 0
        prompt_length = 0
        mapper = {}
        speakers = []
        is_explicit = []
        c_type = {}
        offset_by_chap = {}
        q_type = {}
        valid_quotes = [
            q for q in quotes if q[6] not in ["_group", "_unknowable", "_narr"]
        ]
        
        # First we iterate through the quotes to modify the text and assign the suffixes and prefixes for each quote.
        for idx, quote in enumerate(valid_quotes):
            
            if quote[5] == "Explicit":
                speaker = quote[6]
                end = quote[3] + counter
                start = quote[2] + counter
                
                # 
                text, prompt_length = prefix_suffix_quote_id(
                            text, start, end, idx)

                starts.append(start)
                counter += prompt_length
                ends.append(quote[3] + counter)
                is_explicit.append(1)
            else:
                end = quote[3] + counter
                start = quote[2] + counter
                text, prompt_length = prefix_suffix_quote_id(text, start, end, idx)
                starts.append(start)
                counter += prompt_length
                ends.append(quote[3] + counter)
                is_explicit.append(0)
                
            chap_id = quote[1]
            offset_by_chap[chap_id] = counter

            c_type[idx] = quote[8]
            speakers.append(quote[6])
            q_type[idx] = quote[5]
            
        novel_text = text
        
        
        # print(len(sorted(set(re.findall("\|([\d]+)\|[^\|]+\|\|[\d]+\|\|", text)), key=int)))

        total_max_length = args.max_length
        strides = args.strides
        sp_in_chunk = []
        sp_all = {}
        is_exp = []
        is_exp_all = {}
        to_predict = []
        all_texts = []
        for idx in range(len(chap_info)):
            if (idx > 0) and (idx not in offset_by_chap):
                offset_by_chap[idx] = offset_by_chap[idx - 1]
            elif (idx == 0) and (idx not in offset_by_chap):
                offset_by_chap[idx] = 0
        
        # Then we process chapter by chapter, chunking if needed.
        for num, row in chap_info.iterrows():

            start, end = int(row["titleStartByte"]), int(row["textEndByte"])

            if num == 0:
                end += offset_by_chap[num]
            else:
                start += offset_by_chap[num - 1]
                end += offset_by_chap[num]

            toks = tokenizer.encode(
                novel_text[start:end],
                return_overflowing_tokens=True,
                add_special_tokens=False,
                stride=strides,
                max_length=total_max_length,
            )

            texts = tokenizer.batch_decode(toks)
            all_texts.extend(texts)
            if args.right_context_length == 0 :
                max_left_context = 0
            else : 
                max_left_context = max(0, strides - args.right_context_length - 200)
            
            
            for idx, text in enumerate(texts):
                if len(texts) == 1:
                    text = tokenizer.decode(toks[idx])
                    
                # Striding if needed
                elif idx == 0:

                    if args.right_context_length == 0 :
                        text = tokenizer.decode(toks[idx])
                    else : 
                        text = tokenizer.decode(toks[idx][: -args.right_context_length + 100])


                elif idx == len(texts) - 1:
                        text = tokenizer.decode(toks[idx][max_left_context: ])


                else:
                    if args.right_context_length == 0 :
                        text = tokenizer.decode(toks[idx][max_left_context: ])
                    else : 
                        text = tokenizer.decode(toks[idx][max_left_context : -args.right_context_length+100])

                
                # Then we process each quote ids in the current chunk, find its speaker and add it to the data.
                spp = {}
                iee = {}
                tp = []
                quote_ids = sorted(set(re.findall("\|([\d]+)\|[^\|]+\|\|[\d]+\|\|", text)), key=int)
                for id in quote_ids:
                    if ((args.avoid_long_tail) & (c_type[int(id)] != "minor")) | (
                        not args.avoid_long_tail
                    ):
                        spp[id] = speakers[int(id)]
                        iee[id] = is_explicit[int(id)]
                        if id not in sp_all:
                            sp_all[id] = speakers[int(id)]
                        if id not in is_exp_all:
                            is_exp_all[id] = is_explicit[int(id)]

                        # we only predict non explicit quotes
                        if args.only_nexp:
                            if not is_explicit[int(id)]:
                                if idx > 0:
                                    if args.predict_all : 
                                        tp.append(id)
                                    elif (id not in sum(to_predict, [])) & (id not in tp):
                                            tp.append(id)
                                else:
                                    tp.append(id)
                        else:
                            if idx > 0:
                                # avoid predicting twice.
                                if args.predict_all : 
                                    tp.append(id)
                                elif (id not in sum(to_predict, [])) & (id not in tp):
                                        tp.append(id)
                            else:
                                tp.append(id)

                to_predict.append(tp)
                sp_in_chunk.append(spp)
                is_exp.append(iee)
        candidates = list(set(list(sp_all.values())))

        aliases = {}
        for cand in speaker_info["id2parent"].values():
            # for cand in candidates:
            eid = speaker_info["name2id"][cand]
            c_aliases = speaker_info["id2names"][eid]
            for alias in c_aliases:
                # if alias != cand :
                aliases[alias] = cand

        name2id = speaker_info["name2id"]
        print(len(sp_all))
        data[novel] = {
            "chunks": all_texts,
            "speakers_by_chunk": sp_in_chunk,
            "speakers": sp_all,
            "is_exp_chunk": is_exp,
            "is_explicit": is_exp_all,
            "candidates": candidates,
            "to_predict": to_predict,
            "aliases": aliases,
            "name2id": name2id,
            "quote_type": q_type,
        }
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    with open(args.save_path, "w") as f:
        json.dump(data, f)
