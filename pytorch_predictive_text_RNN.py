import torchtext
from torchtext.data import get_tokenizer

import torch
from torch import nn

import numpy as np

import random
import time
import json


dataset = ["Alice saw Bob.",
           "Alice saw Carmen.",
           "Bob saw Alice.",
           "Bob saw Carmen.",
           "Carmen saw Alice.",
           "Carmen saw Bob."]


with open('predictive_text_datasets.json') as file_object:
        # store file data in object
        datasets = json.load(file_object)
        dataset = datasets["childrens_book"]
        #dataset = datasets["verb_tenses"]
        #dataset = datasets["simple_verb_tenses"]
        #dataset = datasets["countries_and_languages"]


num_epochs = 1000
hidden_size = 256
learning_rate = 0.001
print_interval = 10



tokenizer = get_tokenizer("basic_english")
# tokenize dataset
dataset = [tokenizer(sentence) for sentence in dataset]



vocabulary = []
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

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.sigmoid(self.in2hidden(combined))
        output = self.in2output(combined)
        return output, hidden
    
    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))

    def tensor_guess_to_word(self, tensor_guess):
        prediction_tensor = tensor_guess[0]
        highest_value = 0
        highest_value_index = 0
        for i in range(len(prediction_tensor)):
            if prediction_tensor[i] > highest_value:
                highest_value = prediction_tensor[i]
                highest_value_index = i
        next_word = vocabulary[highest_value_index]
        return next_word


    def tensor_guess_to_probabilities(self, tensor_guess):
        prediction_tensor = tensor_guess[0]
        highest_value = 0
        highest_value_index = 0
        for i in range(len(prediction_tensor)):
            if prediction_tensor[i] > highest_value:
                highest_value = prediction_tensor[i]
                highest_value_index = i
        next_word = vocabulary[highest_value_index]

        probabilities = {}
        for i in range(len(prediction_tensor)):
            probabilities[vocabulary[i]] = round(prediction_tensor[i].item(),2)
        return probabilities
        

    def predict_next_word(self, input_word:str):
        input_word = word_to_one_hot_tensor(input_word)
        output, hidden_state = self(input_word, self.init_hidden())

        print(self.tensor_guess_to_word(output))
        return output,hidden_state

    def predict_rest_of_sentence(self, input_word:str, show_probabilities = False):
            out_string = input_word + "... "
            next_word = input_word
            hidden_state = self.init_hidden()
            while next_word != ".":
                input_word = word_to_one_hot_tensor(next_word)
                output, hidden_state = self(input_word, hidden_state)


                if show_probabilities:
                    print(next_word.upper())
                    print(self.tensor_guess_to_probabilities(output))

                next_word = self.tensor_guess_to_word(output)
                out_string += " "+next_word
                
            print(out_string)
                

model = MyRNN(len(vocabulary), hidden_size, len(vocabulary))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




for epoch in range(num_epochs):
    for sentence in dataset:
        hidden_state = model.init_hidden()
        input_tensor = get_one_hot_sentence_tensor(sentence)

        
        loss = 0
        index = 0
        for word in input_tensor:
            if index+1 >= len(input_tensor):
                break
            output, hidden_state = model(word, hidden_state)
            current_loss = criterion(output, input_tensor[index+1])
            loss += current_loss
            index += 1

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()


    if epoch % print_interval == 0:
        print("Epoch "+ str(epoch) + " loss: "+str(loss.item()))


model.predict_rest_of_sentence("carmen", show_probabilities = True)



