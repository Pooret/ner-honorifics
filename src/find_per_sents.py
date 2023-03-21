# imports

import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import csv

# set folder path to retrieve data
INPUT_PATH = "../input/"
DATA_PATH = "conll-2003/"

print('loading sentences...')

# read in NER dataset as pandas DataFrame
data = pd.read_csv(INPUT_PATH + DATA_PATH + "ner.csv", encoding='latin1')

# extract sentences
sentences = data['text'].tolist()

# load ner spacy model
nlp_spacy_lg = spacy.load('en_core_web_lg')

# create spacy objects from sentences
sentences_spacy_lg = [] # to store spacy objects

for text in tqdm(sentences, desc='converting text into spacy objects'):
    sentence = [] # to hold individual sentences

    for token in nlp_spacy_lg(text):
        sentence.append(token)

     # append sentence to list   
    sentences_spacy_lg.append(sentence)

# function to find sentences with PERSON ner tags from the spacy model
def find_sentences_with_PERSON_tags(sentences):
    samples = []
    for sentence in tqdm(sentences, 'finding sentences with PERSON tags'):
        for token in sentence:
            if token.ent_type_ == 'PERSON' and sentence not in samples:
                samples.append(sentence)
    print('found {} sentences with PERSON tags'.format(len(samples)))        
    return samples    

# call function to create list of lists of words in person sentences
per_words = find_sentences_with_PERSON_tags(sentences_spacy_lg)

# change list of lists of words to list of person sentences
per_sentences = []
for word_list in per_words:
    words = []
    for word in word_list:
        words.append(word.text)
    per_sentences.append(" ".join(words))

# save sentences to csv file
file_name = "PER_sentences.csv"

print('saving sentences to file')

# write the data to a csv file
with open(INPUT_PATH + file_name, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for sentence in per_sentences:
        writer.writerow([sentence])