from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, BertTokenizer
import string
import json 
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd 
import numpy as np 
from argparse import ArgumentParser
import os 
from collections import defaultdict 
import pickle as pkl 
from token_hgface import HGFACE_TOKEN

# device = torch.device("cuda:0")
device = torch.device("mps")
# device  = "cpu"

def verbalize(df, idx, columns=None) :
    text = ""
    row = df.iloc[idx]
    if columns is None : 
        columns = df.columns
    for cnt, column in enumerate(columns) : 
        if cnt < len(df.columns) - 1: 
            text += f"{column}: {row[column]}; "
        else : 
            text += f"{column}: {row[column]}"
    return text

def get_min_k_perc(logits, tokens, k_perc) : 
    
    if isinstance(k_perc, list) : 
        results = {}
        for kk in k_perc: 
            k = int(kk * logits.size(1))
            logsoft =torch.log_softmax(logits[:, :-1], dim = -1).gather(2, tokens[:, 1:].unsqueeze(2)).squeeze(2)
            values, indices = torch.topk(logsoft, k, largest = False)
            results[kk] = (values.sum(1) / values.count_nonzero(dim=1)).tolist()
    elif isinstance(k_perc, float) : 
            k = int(k_perc * logits.size(1))
            logsoft =torch.log_softmax(logits[:, :-1], dim = -1).gather(2, tokens[:, 1:].unsqueeze(2)).squeeze(2)
            values, indices = torch.topk(logsoft, k, largest = False)
            results[k_perc] = (values.sum(1) / values.count_nonzero(dim=1)).tolist()
    # for tok in tokenizer.all_special_ids : 
    #     logsoft[prompts[:,s:] == tok] = 0
    # logsoft[prompts[:,-21:] == 29871] = 0
    # logsoft[prompts[:,-20:] == 259] = 0

    return results


if __name__ == "__main__" : 
    parser = ArgumentParser()
    parser.add_argument("--pdnc_path", help="Path to default PDNC data (from github)", default = "/data/datasets/project-dialogism-novel-corpus/data/")
    parser.add_argument("--save_path", help="Path to save results", default = "results/contamination/")
    parser.add_argument("--perc_examples", help="Number of instances per novel", default = 0.2, type=float)

    args = parser.parse_args()
    

    torch.manual_seed(42)
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HGFACE_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    quantization_config = BitsAndBytesConfig(load_in_8bit=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        is_decoder=True,
        token=HGFACE_TOKEN,
        device_map="auto",
        quantization_config=quantization_config,
    ).eval()
    all_novels = [i for i in os.listdir(args.pdnc_path) if os.path.isdir(os.path.join(args.pdnc_path, i))]

    k_perc = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] 
    seen_min_k_scores = {k:{} for k in k_perc}
    seen_ppl_scores = {}
    pbar = tqdm(all_novels, total=len(all_novels))
    
    for novel in pbar : 
        df = pd.read_csv(os.path.join(args.pdnc_path, novel, "quotation_info.csv"))
        tokens = []
        for num in range(len(df)) : 
            tokens.append(tokenizer.encode(verbalize(df, num), return_tensors="pt"))
        # tokens = torch.cat(tokens)
        tokens = [tok for tok in tokens if all([tok.size(1) > 16, tok.size(1) < 256])]
        
        num_examples = int(len(tokens) * args.perc_examples)
        indices = torch.randperm(len(tokens))[:num_examples]
        
        tokens = [tokens[idx] for idx in indices]
        
        min_k_dict = defaultdict(list)
        ppl_scores = []
        for bs in range(0, len(tokens)) : 
            pbar.set_description(f"{novel} [{bs+1}/{num_examples}]")

            toks = tokens[bs]
            with torch.no_grad() : 
                out = model(toks.to(device), labels=toks.to(device))
            loss, logits = out[:2]
            min_k_scores = (get_min_k_perc(logits.cpu(), toks, k_perc))
            for k, val in min_k_scores.items() : 
                min_k_dict[k].extend(val)
            ppl_scores.append(torch.exp(loss.cpu()).item())  
        # results = get_min_k_perc(tokens, k_perc)
        for k, res in min_k_dict.items() : 
            seen_min_k_scores[k][novel] = res
        seen_ppl_scores[novel] = ppl_scores
    
    if not os.path.exists(args.save_path) : 
        os.makedirs(args.save_path)
        
    with open(os.path.join(args.save_path, "min_k_perc.json"), "wb") as f :
        pkl.dump(seen_min_k_scores, f)
    with open(os.path.join(args.save_path, "ppl.json"), "wb") as f :
        pkl.dump(seen_ppl_scores, f)
    