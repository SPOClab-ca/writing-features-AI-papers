import pickle
import json
import os, sys, time

start_time = time.time()

lookup_table = {}
with open("../../data/S2ORC/20200705v1/full/pdf_parses/pdf_parses_0.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        lookup_table[obj['paper_id']] = obj 

print ("Loaded lookup table of pdf_parses_0.jsonl in {:.4f} seconds".format(
    time.time() - start_time
))
print ("Num of items in lookup table: {}".format(len(lookup_table)))