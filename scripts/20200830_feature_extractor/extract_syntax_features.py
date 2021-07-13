import pandas as pd
import pickle  
import os, sys, time 
import itertools
import numpy as np
import nltk 
from nltk import sent_tokenize 

import spacy
nlp = spacy.load("en_core_web_md")

from utils import timed_func 
from extract_lex_features import prepare_text


def extract_pos_ratio(proc_abstract, proc_bodytext):
    # https://github.com/explosion/spaCy/blob/master/spacy/lang/en/tag_map.py
    pos_list = [
        "ADJ", "ADV", "ADP", "AUX", "CCONJ", 
        "DET", "INTJ", "NOUN", "NUM", "PART", 
        "PRON", "PROPN", "SPACE", "VERB"
    ]
    df_abs = _count_pos_ratio(
        itertools.chain.from_iterable(proc_abstract), 
        pos_list, 
        "pos_abstract")
    
    df_bodytext = _count_pos_ratio(
        itertools.chain.from_iterable(proc_bodytext), 
        pos_list, 
        "pos_bodytext")
    
    return pd.concat([df_abs, df_bodytext], axis=1)


def _count_pos_ratio(nlp_doc, pos_list, feat_prefix=""):
    counts = {}
    for token in nlp_doc:
        if token.pos_ in pos_list:
            if token.pos_ not in counts:
                counts[token.pos_] = 1
            else:
                counts[token.pos_] += 1
    # Normalize
    total = np.sum(list(counts.values()))
    ratios = {}
    for pos in pos_list:
        ratios["{}_{}".format(feat_prefix, pos)] = [counts.get(pos, 0) / total] 
    return pd.DataFrame(ratios)


def extract_voice_ratio(proc_abstract, proc_bodytext):
    df_abs = _count_voice_ratio(proc_abstract, "voice_abstract")
    df_bodytext = _count_voice_ratio(proc_bodytext, "voice_bodytext")
    return pd.concat([df_abs, df_bodytext], axis=1)


def _count_voice_ratio(nlp_docs, feat_prefix=""):
    counters = {"active": 0, "passive": 0, "other": 0}
    for doc in nlp_docs:
        doc_result = "other"
        for token in doc:
            if token.dep_ == "nsubj":
                doc_result = "active"
                break
            if token.dep_ == "nsubjpass":
                doc_result = "passive"
                break
        counters[doc_result] += 1
    # Normalize
    total_sent = np.sum([counters[k] for k in counters])
    result = {}
    for v in counters:
        result["{}_{}".format(feat_prefix, v)] = [counters[v] / total_sent]
    return pd.DataFrame(result)


def spacy_proc(abstract, bodytext_list):
    """
    Input:
        abstract: str
        bodytext_list: list of str (each str is a sentence)
    Output:
        proc_abstract: list (len # sentences) of spacy.tokens.doc.Doc
        proc_bodytext: list (len # sentences) of spacy.tokens.doc.Doc
    """
    proc_abstract = [nlp(sent) for sent in sent_tokenize(abstract)]
    proc_bodytext = [nlp(sent) for sent in bodytext_list]
    return proc_abstract, proc_bodytext


@timed_func
def run_syntax_features_extraction(mds, cat_pdf, args, print_per=1000, checkpoint=None, prev_result=None):
    
    if checkpoint is not None and checkpoint["label"] == "extract_syntax_features":
        all_dfs = checkpoint['all_dfs']
        skipped = checkpoint["skipped"]
        start_time = checkpoint["start_time"]
        i = checkpoint["i"]    
        prev_result = checkpoint["prev_result"]    
    else:
        all_dfs = []
        skipped = 0
        start_time = time.time()
        i = 0

    while i < len(mds):
        if i % 1000 == 0:
            elapsed_time = time.time() - start_time
            print ("Processed {} papers. Skipped {}. Time elapsed: {:.2f}s ({:.2f} hours)".format(i, skipped, elapsed_time, elapsed_time / 3600))
            checkpoint = {
                "label": "extract_syntax_features",
                "i": i,
                "skipped": skipped,
                "start_time": start_time,
                "all_dfs": all_dfs,
                "prev_result": prev_result
            }
            with open(args.checkpoint, "wb") as f:
                pickle.dump(checkpoint, f)

        md = mds[i]
        paper_id = md['paper_id']
        pdfparse = cat_pdf[paper_id]

        abstract, bodytext_list = prepare_text(md, pdfparse)
        if abstract is None:
            i += 1
            skipped += 1
            continue
        proc_abstract, proc_bodytext = spacy_proc(abstract, bodytext_list)
        identifier_df = pd.DataFrame({"paper_id": [paper_id]})
        pos_ratio_features_df = extract_pos_ratio(proc_abstract, proc_bodytext)
        voice_ratio_df = extract_voice_ratio(proc_abstract, proc_bodytext)

        row_df = pd.concat([identifier_df, pos_ratio_features_df, voice_ratio_df], axis=1)
        all_dfs.append(row_df)

        i += 1
        

    result_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    print ("Num. syntax features: {}".format(len(list(result_df))))
    print ("Collected {} items. Skipped {}".format(
        len(result_df), skipped
    ))
    return result_df, prev_result
