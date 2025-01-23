import re 


def restart_quote_seq(text, start_from = 1) : 
    mapper = {}
    all_seq = list(re.finditer(r"(\|[\d]+\|)|(\|\|[\d]+\|\|)", text))
    offset = 0 
    spans = []
    counter  = 0
    for idx, match in enumerate(all_seq) : 
        s,e = match.span()
        val = match.group(1) if match.group(1) is not None else match.group(2)
        v = re.findall("[\d]+",val)[0]
        if v not in mapper : 
            mapper[v] = str(counter + start_from)
            counter += 1
        new_val = re.sub("[\d]+", mapper[v], val)

        text = text[:s + offset] + new_val + text[e + offset:] 

        offset += len(new_val) - len(val)

    return text, mapper

def get_explicit(text, mapper=None) : 
    
    # matches = re.finditer(r"[\|][\d]+[\|][^\|]*\|\|([\d]+)\|\|said by ([\w\.\s]+)\|\|", text)
    matches = re.finditer(r"\|([\d]+)\|\s([^\|]+)\s\|", text)

    out = ""
    for m in matches : 
        speaker = m.group(2)
        quote = m.group(1)
        s,e = m.span()
        if mapper is not None : 
            try : 
                speaker = mapper[speaker]
            except : 
                pass
        out += f"{quote}: {speaker}\n"

    return out

def add_preds_in_context(text, past_preds) : 
    quotes_in_ctx = re.findall(r"\|\|([\d]+)\|\|", text)
    for qid, pred in past_preds.items() : 
        qqid = f"||{qid}||"
        if qid in quotes_in_ctx : 
            text = text.replace(qqid, qqid + f"{pred}|")
    return text

def sub_explicits(text, remove=False) : 
    # return re.sub("said by ([\w\s\.]+)\|\|", "", text)
    if remove : 
        # return re.sub("said by ([\w\s\.\-]+)\|\|", "", text)
        return re.sub("said by ([^\|]+)\|\|", "", text)

    # return re.sub("said by ([\w\s\.]+)\|\|", lambda x: " " + x.group(1) +" ||", text)
    return re.sub("said by ([^\|]+)\|\|", lambda x: " " + x.group(1) +" ||", text)

def sub_pipes(text) : 
    return re.sub("\|(\|[\d]+\|)\|", lambda x: x.group(1), text)

from functools import partial

def get_acc(labels, preds) :
    correct = 0 
    total = 0
    for idx in labels.keys() : 
        if idx in preds.keys() : 
            if preds[idx] == labels[idx] : 
                correct += 1
        total += 1
    return correct / total 
def parse_with_regex(string, mapper, regex) : 
    past_preds = {}
    for m in re.finditer(regex, string) : 
        try : 
            # past_preds[mapper[m.group(1)[1:-1]]] = m.group(2)
            past_preds[mapper[m.group(1)]] = m.group(2).strip()
        except : 
            pass
    return past_preds

def parse_answer_with_content_and_pipes(string, mapper) : 
    past_preds = {}
    # Avoid repetitions here
    qids = set(re.findall("[\|]+([\d]+)[\|]+", string))
    speakers = re.findall("[\|]+[\"]?\s([\w\s\.\-]+)[\"\s\|]?[\n\<]+", string)
    # speakers = re.findall("[\|]+[\"]?\s[\"]?([^\n<\|\"]*)", string)
    if len(qids) == len(speakers) : 
        for qi, s in zip(qids, speakers) : 
            try : 
                # past_preds[mapper[m.group(1)[1:-1]]] = m.group(2)
                past_preds[mapper[qi]] =s.strip()
            except : 
                pass
    return past_preds

def parse_answer_with_content(string, mapper) : 
    past_preds = {}
    qids = set(re.findall("[\|]?([\d]+)[:]?[\|]?", string))
    speakers = re.findall(r'\"[\s]?[,\-]?\s([^\n<\|\"]*)[\n<]+', string)
    if len(qids) == len(speakers) : 
        for qi, s in zip(qids, speakers) : 
            try : 
                # past_preds[mapper[m.group(1)[1:-1]]] = m.group(2)
                past_preds[mapper[qi]] =s.strip()
            except : 
                pass
    return past_preds


# parse_markdown = partial(parse_with_regex, regex="\|[\s]?([\d]+)[\s]?\|[\s]?([\w\s\.\-]+)[\s]?\|")
parse_markdown = partial(parse_with_regex, regex="\[\|]+[\s]?([\d]+)[\s]?[\|]+[\s]?([^\n<\"\|]*)[\|]?[\n<]+")
# parse_default = partial(parse_with_regex, regex="[\|\[]?([\d]+)[\|\]]?[:]?\s[\"]?([\w\s\.\-]+)[\"]?[\n\<\]]+")
# parse_default = partial(parse_with_regex, regex="[\|\[]?([\d]+)[\|\]?[:\.]??\s[\"]?([^\"\n<\]]+)[\"]?[\n\<\]]+")
parse_default = partial(parse_with_regex, regex="[\|\[]?([\d]+)[\|\]]?[^\n\d\s]+\s[<]?([^\d\n\">\]]+)[\">]?[\n\<\]]+")

# parse_default = partial(parse_with_regex, regex="[\|]?([\d]+)[\|]?[:\s\"]+([^\n<\"\|]*)")
# parse_default = partial(parse_with_regex, regex="[\|]?([\d]+)[\|]?[:]?\s[\"]?([^\n<\"\|]*)[\n<]+")
parse_python_list =  partial(parse_with_regex, regex="[\[]?([\d]+):\s([^\d\n\">\]\,]+)[\n<\,]+?")
# parse_pipes = partial(parse_with_regex, regex="[\|]+([\d]+)[\|]+[\d]?[:]?\s[\"]?([\w\s\.\-]+)[\"]?[\n\<]+")
parse_pipes = partial(parse_with_regex, regex="[\|]+([\d]+)[\|]+[\d]?[:\s]+[\"]?([^\n<\"\|]*)[\n\<]+")
# parse_stars = partial(parse_with_regex, regex="[\*]?([\d]+)[:\s]+([\w\s\.\-]+)[\*]+[\n\<]+")
parse_stars = partial(parse_with_regex, regex="[\*]+([\d]+)[:\s]+([^\n<\"\|\*]*)")
parse_points = partial(parse_with_regex, regex="[\|\[]?([\d]+)[\|\]]?\.\s([^\n<]+)[\n\<\]]")
PIPELINE = {
    "list_parser": parse_python_list,
    "default_parser": parse_default,
    "markdown_parser": parse_markdown,
    "pipe_parser": parse_pipes,
    "stars_parser": parse_stars,
    "point_parser": parse_points,
    "quote_content_parser": parse_answer_with_content,
    "quote_content_pipe_parser": parse_answer_with_content_and_pipes
}


def parse_model_answer(string, mapper, pipeline=PIPELINE, debug=False) : 
    for pipe_name, pipe in pipeline.items() : 
        past_preds = pipe(string,mapper) 
        if len(past_preds) > 0 :
            if debug: 
                print(f"Parsed with {pipe_name}")
            return past_preds
    print("Error in FORMAT")
    return {}