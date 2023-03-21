# imports
import csv

# specify the directory of the input data
INPUT_PATH = '../input/'

# load sentences from file
sentences = []
with (open(INPUT_PATH + 'PER_sentences.csv', 'r', encoding='utf-8')) as f:
    reader = csv.reader(f)

    for row in reader:
        sentences.append(row)

# batch sentences for NER labeling
# 200 sentences per batch
span_start = 0
span_end = 199

# starting batch number
batch = 0

# number of times batch goes into 200
while batch <= len(sentences) // 200:

    # write sentences to file: batched_sentences
    with open(INPUT_PATH + f'batched_sentences/sentences_{batch}.txt', 'w') as f:
        for sentence in sentences[span_start:span_end]:
            f.write("%s\n" % sentence)

    # update iterables
    batch += 1
    span_start += 200
    span_end += 200