import torch
from torch.utils.data import Dataset

# function to convert tags to tokenized-tags
def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword.
    """
    
    tokenized_sentence = []
    labels = []
    
    sentence = sentence.strip()
    
    for word, label in zip(sentence.split(), text_labels.split()):
        
        # tokenize word and count # of subword tokens
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        
        # add tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)
        
        # add label and multiply by subword length
        labels.extend([label]*n_subwords)

def sigmoid(x):
  """
  Function to constrain logits
  """
  return 1/(1+np.exp(-x))
        
# dataset class for sentences
class dataset(Dataset):
        def __init__(self, data, tokenizer, max_len):
            self.len = len(data)
            sentences, labels = zip(*data)
            self.sentences = sentences
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
          return self.len
            
        def __getitem__(self, index):
            sentence = self.sentences[index]
            labels = self.labels[index]
            tokenized_sentence, labels = tokenize_and_preserve_labels(
                sentence, labels, self.tokenizer)
            
            # add special tokens
            tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
            
            # add Out of label tokens for special tokens
            labels.insert(0, "O") 
            labels.insert(-1, "O")
            
            trunc = self.max_len
            
            # truncate
            if len(tokenized_sentence) > trunc:
                tokenized_sentence = tokenized_sentence[:trunc]
                labels = labels[:trunc]
            else:
                # padding
                tokenized_sentence = tokenized_sentence + ["[PAD]" for _ in range(trunc - len(tokenized_sentence))]
                labels = labels + ["O" for _ in range(trunc - len(labels))]

            attn_mask = [1 if token != "[PAD]" else 0 for token in tokenized_sentence]
            ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
            label_ids = [labels_to_ids[label] for label in labels]
            
            return {
                "ids" : torch.tensor(ids, dtype=torch.long),
                "mask" : torch.tensor(attn_mask, dtype=torch.long),
                "targets" : torch.tensor(label_ids, dtype=torch.long)
            }