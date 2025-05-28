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

# def parse_answer(output) : 
#     out = re.findall(r"<name>([^<>]+)[<>](/name>>)?",output)
#     if len(out) > 0 : 
#         return out[0][0]
#     else :
#         out = re.findall(r": ([^<]+)[<>]?", output)
        
#     if len(out) == 0 : 
#         print(f"Error with answer: {output}")
#         return None
#     else : 
#         return out[0]

def parse_answer(output) : 
    out = re.findall(r"<name>[\[]?([^<>\]]+)[\]]?[<>]",output)
    if len(out) > 0 : 
        return out[0]
    else :
        out = re.findall(r"<name>[\[]?([^<>\]]+)[\]]?[<>]",output)
    if len(out) > 0 : 
        return out[0]
    else : 
        out = re.findall(r"<name>[\[]?([^<>\]]+)[\]]?[<>]",output)
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
                        
    
@torch.no_grad()
def predict_unofficial(passage, print_prompt = False):
    text="""You have seen the following passage in your training data. What is the proper name that fills in the [MASK] token in it?  This name is exactly one word long, and is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain."""

    text += '\n\n\"Stay gold, [MASK], stay gold.\"'
    prompts =  [{"role":"user", "content" : text}]
    prompts.append({"role":"assistant", "content" : "<name>Ponyboy</name>"})
    prompts.append({"role":"user", "content" : '\"The door opened, and [MASK], dressed and hatted, entered with a cup of tea.\"'})
    prompts.append({"role":"assistant", "content" : "<name>Gerty</name>"})
    prompts.append({"role":"user", "content" : f'\"{passage}\"'})

    ids = tokenizer.apply_chat_template(prompts, return_tensors="pt", add_generation_prompt=True).to(device)
    size = ids.size(1)
    if print_prompt : 
        print(tokenizer.batch_decode(ids)[0])
    return tokenizer.batch_decode(model.generate(ids, do_sample=False, max_new_tokens=50)[:,size:])[0]


@torch.no_grad()
def predict(passage, print_prompt = False):
    text="""You have seen the following passage in your training data. What is the proper name that fills in the [MASK] token in it?  This name is exactly one word long, and is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.   
    
You must format your answer in <name>[NAME]</name> tags.

Example:

Input: Stay gold, [MASK], stay gold.
Output: <name>Ponyboy</name>

Input: The door opened, and [MASK], dressed and hatted, entered with a cup of tea.
Output: <name>Gerty</name>

Input: %s
Output: 

""" % passage

    prompts =  [{"role":"user", "content" : text}]
    # prompts.append({"role":"assistant", "content" : "<name>Ponyboy</name>"})
    # prompts.append({"role":"user", "content" : '\"The door opened, and [MASK], dressed and hatted, entered with a cup of tea.\"'})
    # prompts.append({"role":"assistant", "content" : "<name>Gerty</name>"})
    # prompts.append({"role":"user", "content" : f'\"{passage}\"'})

    ids = tokenizer.apply_chat_template(prompts, return_tensors="pt", add_generation_prompt=True).to(device)
    size = ids.size(1)
    if print_prompt : 
        print(tokenizer.batch_decode(ids)[0])
    return tokenizer.batch_decode(model.generate(ids, do_sample=False, max_new_tokens=50)[:,size:])[0]

if __name__ == "__main__" : 
    parser = ArgumentParser()
    parser.add_argument("--data_path", help="Path to data", default="../data/pdnc_source/")
    parser.add_argument("--num_examples", help="Number of passages per novel", default=100, type=int)
    parser.add_argument("--result_path", help="Path to result data", default="results/name_cloze/")
    parser.add_argument("--novel_path", help="Path to a single novel", default=None)

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
    print(novels)
    for idx, novel in enumerate(novels) : 
        novel_path = paths[idx]
        data = read_data(os.path.join(novel_path, f"{novel}.name_cloze.txt"))
        correct = 0
        total = 0
        preds = []
        golds = []
        indices = np.random.permutation(range(len(data)))[:args.num_examples]
        pbar = tqdm(enumerate([data[idx] for idx in indices]), total=len(indices))

        for num, d in pbar : 
            gold = d[5]
            passage = d[6]
            if num < 5 : 
                out = predict(passage, print_prompt=True)
            else : 
                out = predict(passage)
            
            
            pred = parse_answer(out) 
            print(out, "\t", pred, "\t", gold)
            preds.append(pred)
            golds.append(gold)
            if pred == gold : 
                correct += 1 
            total += 1 
            pbar.set_description(f"{novel}: {(correct / total) : 0.2f}")

        novel_results = {
                "accuracy": correct / total,
                "predictions": preds,
                "labels": golds
            }
        with open(os.path.join(args.result_path, f"{novel}.res"), "w") as f : 
            json.dump(novel_results, f)