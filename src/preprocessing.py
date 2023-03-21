# imports
import os
import re
import json
import numpy as np
import spacy
from tqdm import tqdm
import pickle
import csv

# directories and parameters
INPUT_PATH = "../input/"
directory = os.path.join(INPUT_PATH, "annotated_sentences")

# blank model to create doc objects
nlp = spacy.blank('en')

# load annotated sentences
annotated_sentences = {}
for root, dirs, files in os.walk(directory):
    for filename in files:
        
        # get batch number from "annotations_{batch}.json" )
        batch = int("".join(re.findall(r"[0-9]", filename)))
        
        # open and load json files
        f = open(os.path.join(root, filename))
        json_files = json.load(f)
        
        # store each sentence in a dictionary
        sentence_batch = {}
        
        # enumerate through each sentence in the json file
        for i, sentence in enumerate(json_files['annotations']):
            
            # sentences stored in a dictionary
            text_ents = {}
            text = sentence[0]
            ents = sentence[1]['entities']
            
            # entites stored in a list
            ents_list = []
            for ent in ents:
                span = {'start':ent[0], 'end':ent[1], 'label':ent[2]}
                ents_list.append(span)
            
            # keys:
            # text -> raw sentence
            # ents -> named entities and their respective spans in the sentence 
            # title -> batch and sentence number
            text_ents['text'] = text
            text_ents['ents'] = ents_list
            text_ents['title'] = [f"batch_{batch}_sentence_{i}"]
            
            # add each sentence dict to dictionary containing all sentences per batch
            sentence_batch[i] = text_ents
        
        # add sentence batch to annonated_sentences dict
        annotated_sentences[batch] = sentence_batch
        # close the json file containing the sentences for that batch
        f.close()

# create a tokenized dict
tokenized_entities = {}

# go through each sentence batch
for batch in annotated_sentences.keys():
    
    # go through all sentences per batch
    for n in annotated_sentences[batch].keys():
        
        # dicionary storing entity spans
        text_ents = annotated_sentences[batch][n]
        
        # create a list to store spans
        spans = []
        
        # create doc object to convert char spans to token ids
        doc = nlp(text_ents['text'])
        for span in text_ents['ents']:
            spans.append(doc.char_span(span['start'], span['end'], span['label']))
            
        # create Named Entity Tokens and word tokens
        # The "O" represents tags that are out of span
        NER_tokens = ["O" for element in doc]
        tokens = [word for word in doc]
        
        # go through the spans and assign named entities
        for span in spans:
            
            # "B" starts the beginning of tags entity
            NER_tokens[span.start] = "B-" + span.label_
            i = span.start + 1
            
            # "I" for Inbetween tagged entities
            while i < span.end:
                NER_tokens[i] = "I-" + span.label_
                i += 1
                
        # add tokens to dict
        tokenized_entities[batch, n] = (doc.text, spans, NER_tokens)

# write to .csv file
#--------------------------------
# header for csv file
header = ['sentence', 'tags']

with open(INPUT_PATH + "bert_tokenized_sents.csv", \
          "w", encoding='UTF-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for batch in range(len(annotated_sentences.keys())):
        for n in annotated_sentences[batch].keys():
            text, entities, tokens = tokenized_entities[batch, n]
            writer.writerow([text, " ".join(tokens)])