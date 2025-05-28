import re 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import json 
from argparse import ArgumentParser
import os 
import numpy as np 
import pandas as pd 
import torch 
import sys 
from token_hgface import HGFACE_TOKEN
import pickle 
from process_pdnc_chunks import prune_info, get_enhanced_char_list

torch.manual_seed(42)
np.random.seed(42)
# device = torch.device("cuda:0")
device = torch.device("mps")

def clean_context(text) : 
    text = re.sub(r"([\w\d\.\,\:])[\n]{1}([\w\d\.\,\:])", lambda x: x.group(1) + " " + x.group(2), text)
    text = re.sub(r"_" , " ", text)
    text = re.sub(r"[ ]{3}" , " ", text)
    text = re.sub("\n$", "", text)
    return text

def parse_answer(output) : 
    out = re.findall(r"<speaker>[\[]?([^<>\]]+)[\]]?[<>]",output)
    if len(out) > 0 : 
        return out[0]
    else :
        out = re.findall(r"<Speaker>[\[]?([^<>\]]+)[\]]?[<>]",output)
    if len(out) > 0 : 
        return out[0]
    else : 
        out = re.findall(r"<SPEAKER>[\[]?([^<>\]]+)[\]]?[<>]",output)
    if len(out) > 0 : 
        return out[0]
    else :
        out = re.findall(r": ([^<]+)[<>]?", output)
        
    if len(out) == 0 : 
        print(f"Error with answer: {output}")
        return None
    else : 
        return out[0]

def read_data(filename) : 
    data = []
    with open(filename,"r") as f : 
        for line in f.readlines() : 
            data.append(tuple(line.split("\t")))
    return data 
                        


def iterate(tokens, target_byte, target_col = "byte_onset") : 
    try : 
        tok = tokens[tokens[target_col] == target_byte].iloc[0].name
    except : 
        tok = None
    cnt = 0 
    while tok is None: 
        tokk = tokens[tokens[target_col] == target_byte-cnt]
        if tokk.shape[0] == 0 :
            tokk = tokens[tokens[target_col] == target_byte+cnt]
            if tokk.shape[0] == 0 :
                cnt += 1 
            else : 
                tok = tokk.iloc[0].name
        else : 
            tok = tokk.iloc[0].name

    return tok 
def find_ner(entities, start_tok, end_tok) : 
    u = entities.start_token.apply(lambda x: x >= start_tok)
    v = entities.end_token.apply(lambda x: x <= end_tok)
    res = u == v
    res[res == True]
    return entities.loc[res]


def get_corrupted_passages(df, tokens, entities, id2rep, context_sentences = 5) : 
    iterator = tqdm(df.iterrows())
    
    data = []
    cnt_ex = 0 
    cnt_an = 0
    for idx, row in iterator : 
        # if row["qType"] == "Anaphoric" : 

        idd = row["qID"]
        q_type = row["qType"] 
        if row["speakerType"] != "minor" : 
            # if row["qType"] == "Implicit" : 
                if row["speakerID"] in id2rep : 

                    s_id = row["speakerID"] 
                    s_aliases = id2rep[s_id]
                    # s_gender = p_char_info["id2gender"][s_id] 

                    sb, eb = eval(row["qSpan"])

                    st = iterate(tokens, sb)
                    et = iterate(tokens, eb)
                    if any([st is None, et is None]) : 
                        continue

                    try : 
                        st_sid = max(tokens.loc[st].sentence_ID - context_sentences, 0)
                        st = tokens[tokens.sentence_ID == st_sid].index[0]

                        et_sid = min(tokens.loc[et].sentence_ID + context_sentences, max(tokens.sentence_ID))
                        et = tokens[tokens.sentence_ID == et_sid].index[-1]
                    except : 
                        continue 

                    sub_entities = find_ner(entities, st, et)
                    prop_entities = sub_entities[sub_entities.prop == "PROP"]

                    # pron_entities =  
                    try : 
                        sub = tokens.loc[st:et]

                        sb_ = sub.iloc[0].byte_onset
                        eb_ = sub.iloc[-1].byte_offset
                        passage = text[sb_ : eb_] 
                        true_passage = passage
                        
                    except : 
                        continue 

                    # Mask the referring name if explicit quote
                    if (row["qType"] == "Explicit") : 
                        if isinstance(row["refExp"], str) : 
                            if row["refExp"] in passage : 
                                match = next(re.finditer(row["refExp"], passage))
                                match_s, match_e = match.span()
                                match_s, match_e = match_s + sb_, sb_ + match_e
                                match_st = iterate(tokens, match_s)
                                match_et = iterate(tokens, match_e)
                                entity = find_ner(prop_entities, match_st, match_et)
                                if entity.shape[0] == 1 : 
                                    truth = entity.iloc[0].text
                                    m_st, m_et = entity.iloc[0].start_token, entity.iloc[0].end_token
                                    m_sb, m_eb = tokens.iloc[m_st].byte_onset, tokens.iloc[m_et].byte_offset                            
                                    passage_byte_onset = m_sb - sb_
                                    passage_byte_offset = m_eb - sb_
                                    passage = passage[:passage_byte_onset] + "[MASK]" + passage[passage_byte_offset:] 
                                else : 
                                    continue
                    if (row["qType"] == "Explicit") & ("[MASK]" not in passage ): 
                        continue 
                    cnt = 0 
                    corrupted_passage = passage 
                    s_aliases = list(s_aliases.items())

                    s_aliases.sort( key=lambda l: len(l[0]), reverse=True)
                    s_aliases = dict(s_aliases)
                    offset = 0
                    target_quote = row["qText"]
                    for alias, rep in s_aliases.items() :
                        if alias in corrupted_passage : 
                            cnt += 1 
                            corrupted_passage = corrupted_passage.replace(alias, rep)
                        if alias in target_quote : 
                            target_quote = target_quote.replace(alias, rep)
        #                     sub_alias = prop_entities[prop_entities["text"]==alias]

        #                     for n, s_row in sub_alias.iterrows() : 
        #                         st, et = s_row["start_token"], s_row["end_token"]
        #                         sub_m = tokens.loc[st:et]
        #                         sub_m = sub_m[~sub_m["word"].isin(PREFIXES)]
        #                         sb, eb = sub_m.iloc[0]["byte_onset"] - sb_ - offset, sub_m.iloc[-1].byte_offset - sb_ - offset
        #                         corrupted_passage = corrupted_passage[:sb] + mapper_name[s_gender] + corrupted_passage[eb:]
        #                         offset = len(passage) - len(corrupted_passage)
        #                     prefixes = [i for i in PREFIXES if i in alias]
        #                     if len(prefixes) > 0 : 
        #                         longest_prefix = sorted(prefixes, key=lambda x: len(x), reverse=True)[0]
        #                         to_replace = alias.replace(longest_prefix, "").strip()

        #                     corrupted_passage = corrupted_passage.replace(to_replace, mapper_name[s_gender])

                    if cnt > 0 :
                        data.append((row["qType"], corrupted_passage, target_quote, list(s_aliases.keys()), list(s_aliases.values()), true_passage))
    return data

@torch.no_grad()
def predict(passage, target_quote, novel, author, q_type, print_prompt = False):
    prompts = []
    if q_type != "Explicit" : 
        text=f"""You will be given a passage of the book {novel} written by {author} that you have seen in your training data. Find the true speaker name of the target quote. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain. Do not explain your reasoning."""

    else : 
        text=f"""You will be given a passage of the book {novel} written by {author} that you have seen in your training data. Find the proper name that fills the [MASK] token. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain. Do not explain your reasoning."""

    text += "\n\nYou must format your answer in <speaker>[SPEAKER]<\speaker> tags."
    prompts.append({"role":"user", "content" : text + f"""\n\nPassage:
{passage}

Target quote:
\"{target_quote}\""""})

    ids = tokenizer.apply_chat_template(prompts, return_tensors="pt", add_generation_prompt=True).to(device)
    size = ids.size(1)
    if print_prompt : 
        print(tokenizer.batch_decode(ids)[0])
    return tokenizer.batch_decode(model.generate(ids, do_sample=False, max_new_tokens=50)[:,size:])[0]

if __name__ == "__main__" : 
    parser = ArgumentParser()
    parser.add_argument("--data_path", help="Path to data", default="../data/test_pdnc_source/")
    parser.add_argument("--novel_path", help="Path to data", default=None)
    parser.add_argument("--num_quotes", help="Number of passages per novel", default=100, type=int)
    parser.add_argument("--context_sentences", help="Path to data", default=10, type=int)
    parser.add_argument("--result_path", help="Path to result data", default="results/speaker_cloze/")

    args = parser.parse_args()
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HGFACE_TOKEN,
    )

    quantization_config = BitsAndBytesConfig(load_in_8bit=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        is_decoder=True,
        token=HGFACE_TOKEN,
        device_map="auto",
        quantization_config=quantization_config,
    ).eval()
    
    if not os.path.exists(args.result_path) : 
        os.makedirs(args.result_path)
        
    if not args.novel_path : 
        novels = [i for i in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, i ))]
        paths = [os.path.join(args.data_path, novel) for novel in novels]
    else : 
        novels = [os.path.split(args.novel_path)[-1]]
        paths = [args.novel_path]
    
    with open("../data/novel_metadata.json") as f:
        novel_info = json.load(f)
        
    for idx, novel in enumerate(novels) : 
        
        novel_name = novel_info[novel]["novel_name"]
        novel_author = novel_info[novel]["author"]

        novel_path = paths[idx]
        
        df = pd.read_csv(os.path.join(novel_path, "quote_info.csv"))
        text = open(os.path.join(novel_path, "novel.txt")).read()
        tokens = pd.read_csv(os.path.join(novel_path, f"{novel}.tokens"), sep="\t")
        entities = pd.read_csv(os.path.join(novel_path, f"{novel}.entities"), sep="\t")
        entities = entities[entities.cat == "PER"]

        def convert(item) : 
            try : 
                return int(item)
            except : 
                return item

        tokens.byte_offset = tokens.byte_offset.apply(lambda x: convert(x) )
        tokens.byte_onset = tokens.byte_onset.apply(lambda x: convert(x) )

        char_info = pickle.load(open(os.path.join(novel_path, "charInfo.dict.pkl"),"rb"))
        speakers = df[df.speakerType != "minor"].speaker.unique()
        p_char_info = prune_info(char_info, speakers)
        p_char_info = get_enhanced_char_list(p_char_info)

        id2rep = pickle.load(open(os.path.join(novel_path, "id2replacements.pkl"),"rb"))
        
        # df = df.iloc[:100]
        
        data = get_corrupted_passages(df,tokens, entities, id2rep, context_sentences = args.context_sentences)
        
        
        data_e = [i for i in data if i[0] == "Explicit"]
        idx = np.arange(len(data_e))
        np.random.shuffle(idx)
        data_e = [data_e[i] for i in idx[:args.num_quotes]] 

        data_a = [i for i in data if i[0] == "Anaphoric"]
        idx = np.arange(len(data_a))
        np.random.shuffle(idx)
        data_a = [data_a[i] for i in idx[:args.num_quotes]] 

        data_i = [i for i in data if i[0] == "Implicit"]
        idx = np.arange(len(data_i))
        np.random.shuffle(idx)
        data_i = [data_i[i] for i in idx[:args.num_quotes]] 

        
        all_results  = {}
        for exp_name, data in zip(["explicit", "anaphoric", "implicit"], [data_e, data_a, data_i]): 
            
            correct_reasoning = 0
            correct_uncor = 0
            total = 0
            correct_mem = 0 
            
            preds_uncor = []
            preds = []
            golds = []
            golds_corrupted = []
            pbar = tqdm(enumerate(data), total=len(data), desc=f"[{exp_name.upper()}] {novel}")

            for idx, d in pbar : 
                gold = d[3]
                gold_corrupted = d[4]
                passage = d[1]
                target_quote = d[2]
                q_type = d[0]
                true_passage = d[5]
                
                if idx == 0 : 
                    out = predict(passage, target_quote, novel_name, novel_author, q_type, print_prompt=True)
                    # out_uncor = predict(true_passage, target_quote, novel_name, novel_author, q_type, print_prompt=True)
                else : 
                    out = predict(passage, target_quote, novel_name, novel_author, q_type, print_prompt=False)
                    # out_uncor = predict(true_passage, target_quote, novel_name, novel_author, q_type, print_prompt=False)
                
                print(out)
                pred = parse_answer(out) 
                # pred_uncor = parse_answer(out_uncor)
                
                print(pred, "\t", gold, "\t", gold_corrupted)
                # print(pred_uncor, "\t", gold, "\t", gold_corrupted)

                preds.append(pred)
                # preds_uncor.append(pred_uncor)
                golds.append(gold)
                golds_corrupted.append(gold_corrupted)
                
#                 if pred_uncor in gold : 
#                     correct_uncor += 1
                if pred in gold  : 
                    print(passage)
                    correct_mem += 1 
                elif pred in gold_corrupted : 
                    correct_reasoning += 1 
                total += 1 

                # try : 
                #     uncor_acc = round(((correct_uncor) / (total)),2)
                # except : 
                #     uncor_acc = "none"
                try : 
                    reason_acc = round(((correct_reasoning) / (total)),2)
                except : 
                    reason_acc = "none"
                try : 
                    mem_acc = round(((correct_mem ) / (total)),2)
                except : 
                    mem_acc = "none"

                # pbar.set_description(f"[{exp_name.upper()}] {novel}: [ACC] {uncor_acc} - [ReasonACC] {reason_acc} - [MemACC] {mem_acc}")
                pbar.set_description(f"[{exp_name.upper()}] {novel}: [ReasonACC] {reason_acc} - [MemACC] {mem_acc}")

            novel_results = {
                    # "uncor_accuracy": uncor_acc,
                    "reason_accuracy" : reason_acc, 
                    "mem_accuracy" : mem_acc,
                    "predictions": preds,
                    "labels": golds,
                    "corrupted_labels" : golds_corrupted
                }
            all_results[exp_name] = novel_results
            
        with open(os.path.join(args.result_path, f"{novel}.res"), "wb") as f : 
            pickle.dump(all_results, f)