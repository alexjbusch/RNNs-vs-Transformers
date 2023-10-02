



import torchtext
from torchtext.data import get_tokenizer

import json
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt







dataset = ["Alice saw Bob.",
           "Alice saw Carmen.",
           "Bob saw Alice.",
           "Bob saw Carmen.",
           "Carmen saw Alice.",
           "Carmen saw Bob."]


with open('predictive_text_datasets.json') as file_object:
        # store file data in object
        datasets = json.load(file_object)
        #dataset = datasets["childrens_book"]
        #dataset = datasets["verb_tenses"]
        #dataset = datasets["simple_verb_tenses"]
        dataset = datasets["countries_and_languages_medium"]


num_epochs = 500
hidden_size = 256
learning_rate = 0.001
print_interval = 10



tokenizer = get_tokenizer("basic_english")
# tokenize dataset
dataset = [tokenizer(sentence) for sentence in dataset]



vocabulary = []
print(dataset)
for sentence in dataset:
    for word in sentence:
        if word not in vocabulary:
            vocabulary.append(word)

print(vocabulary)


# encode dataset
for i in range(len(dataset)):
    for j in range(len(dataset[i])):
        dataset[i][j] = vocabulary.index(dataset[i][j])



def get_one_hot_sentence_tensor(sentence:list):
    out_tensor = torch.zeros(len(sentence), 1, len(vocabulary))
    for i in range(len(sentence)):
        out_tensor[i][0][sentence[i]] = 1
        
    return out_tensor

def word_to_one_hot_tensor(word:str):
    out_tensor = torch.zeros(1, len(vocabulary))
    index = vocabulary.index(word)
    out_tensor[0][index] = 1
    return out_tensor

def one_hot_tensor_to_word(tensor):
    out_word = ""
    for i in range(len(tensor[0])):
        if tensor[0][i] == 1:
            out_word = vocabulary[i]
    return out_word



print(dataset)




class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        # returns length of dataset (how many elements in dataset)
        return len(self.dataset)

    def __getitem__(self, idx):
        # returns sentence tensor
        return get_one_hot_sentence_tensor(self.dataset[idx])




"""
train_features = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
img = train_features[0].squeeze()
plt.imshow(img, cmap="gray")
plt.show()
"""




from mingpt.model import GPT
model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257 # openai's model vocabulary
model_config.block_size = 1024  # openai's model block_size (i.e. input context length)
model = GPT(model_config)


# your subclass of torch.utils.data.Dataset that emits example
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
train_dataset = CustomDataset(dataset)

from mingpt.trainer import Trainer
train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # many possible options, see the file
train_config.max_iters = 1000
train_config.batch_size = 32
trainer = Trainer(train_config, model, train_dataset)
trainer.run()
