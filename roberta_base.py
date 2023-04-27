# -*- coding: utf-8 -*-
"""roberta-base.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EvGIJvRK-MmsCeVvLxmdMg3iik3lKHt0

# Python and SageMaker Setup
"""

!pip install sagemaker

"""# New Section"""

import numpy as np                                # For matrix operations and numerical processing
import pandas as pd                               # For munging tabular data
import matplotlib.pyplot as plt                   # For charts and visualizations
from IPython.display import Image                 # For displaying images in the notebook
from IPython.display import display               # For displaying outputs in the notebook
from time import gmtime, strftime                 # For labeling SageMaker models, endpoints, etc.
import sys                                        # For writing outputs to notebook
import math                                       # For ceiling function
import json                                       # For parsing hosting outputs
import os                                         # For manipulating filepath names
import sagemaker                                  # Amazon SageMaker's Python SDK provides many helper functions
from sagemaker.predictor import csv_serializer    # Converts strings for HTTP POST requests on inference
from tqdm import tqdm

!pip install awscli



# bucket = sagemaker.Session().default_bucket()
# prefix = 'sagemaker/amazon_fine_food_reviews'
 
# # Define IAM role
# import boto3
# import re
# from sagemaker import get_execution_role

# role = get_execution_role()
# region = boto3.Session().region_name 
# smclient = boto3.Session().client('sagemaker')

from google.colab import drive 
drive.mount('/content/gdrive')

"""# Load Data """

df = pd.read_csv('gdrive/Shareddrives/CIS519/Reviews.csv')
print(df.shape)
df = df.sample(5000, random_state=42)
print(df.shape)

import boto3
import re

def clean_text(line):
    line = re.sub(r'-+',' ',line)
    line = re.sub(r'[^a-zA-Z, ]+'," ",line)
    line = re.sub(r'[ ]+'," ",line)
    line += "."
    return line

# retain necessary columns
columns_to_use = ["Score", "Summary", "Text"]
df = df[columns_to_use]

# convert score to contextual labels
score2labels = {1: "very negative", 2: "negative", 3: "neutral", 4: "positive", 5: "very positive"}
df["Labels"] = df["Score"].apply(lambda x : score2labels[x])
df["Score"] = df["Score"] - 1

# clean text
df["Summary"] = df["Summary"].astype(str)
df["Summary"] = df["Summary"].apply(clean_text)
df["Text"] = df["Text"].astype(str)
df["Text"] = df["Text"].apply(clean_text)

# remove comments that are smaller than 20 words
df = df[df['Text'].apply(lambda x: len(x.split(" "))) >= 20]
df = df.reset_index(drop=True)
df.head()

from sklearn.model_selection import train_test_split

# Split the data into train and test sets (80/20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Split the train data into train and validation sets (75/25 split)
train_df, valid_df = train_test_split(train_df, test_size=0.25, random_state=42)

# Print the resulting sizes of each set
print("Training set size:", len(train_df))
print("Validation set size:", len(valid_df))
print("Test set size:", len(test_df))

"""# BERT for sequence classification"""

train_batch_size = 32
val_batch_size = 32
test_batch_size = 32
epochs = 5
seed = 42
learning_rate = 2e-5

feature = "text"
feature = "summary"
base_model = "roberta-base"
model_name = feature + "_" + base_model
print(model_name)

!pip install transformers

import torch
from torch import nn
from transformers import (BertTokenizer, BertForSequenceClassification,
                          RobertaTokenizer, RobertaForSequenceClassification,
                          DistilBertTokenizer, DistilBertForSequenceClassification,
                          set_seed, AdamW, get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, random_split
from torch.nn import CrossEntropyLoss

set_seed(seed)

# Load the tokenizer and model
if base_model == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(base_model)
    model = BertForSequenceClassification.from_pretrained(base_model, 
                                                          num_labels=5)
elif base_model == "roberta-base":
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    model = RobertaForSequenceClassification.from_pretrained(base_model, 
                                                              num_labels=5)
elif base_model == "distilbert-base-uncased":
    tokenizer = DistilBertTokenizer.from_pretrained(base_model)
    model = DistilBertForSequenceClassification.from_pretrained(base_model,
                                                                 num_labels=5)
else:
    raise ValueError("Invalid base_model value. Supported models: 'bert-base-uncased', 'roberta-base', 'distilbert-base-uncased'")

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)

from sklearn.metrics import accuracy_score, f1_score

# Function to calculate the accuracy of our predictions vs labels
def flat_matrices(preds, labels):
    preds = nn.functional.softmax(preds, dim=1).squeeze().to('cpu').numpy().reshape(-1, 5)
    preds = np.argmax(preds, axis=1)
    labels_flat = labels.flatten()
    return accuracy_score(preds, labels_flat), f1_score(preds, labels_flat, average='weighted')

def prepare_input(text):
    inputs = tokenizer(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=256,
        padding='max_length', 
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

class custom_dataset(Dataset):
    def __init__(self, df, feature):
        if (feature == "text"):
            self.text = df['Text'].values
        elif (feature == "summary"):
            self.text = df['Summary'].values

        self.label = df['Score'].values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = prepare_input(self.text[item])
        target = torch.tensor(self.label[item], dtype=torch.long)
        return inputs, target

train_dataset = custom_dataset(train_df, feature)
train_dataloader = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset),
    batch_size=train_batch_size,
)    

val_dataset = custom_dataset(valid_df, feature)
val_dataloader = DataLoader(
    val_dataset,
    sampler = SequentialSampler(val_dataset),
    batch_size=val_batch_size,
)

total_steps = len(train_dataloader) * epochs
warmup_steps = int(total_steps * 0.2)

optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = 1e-8
                )

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps,
                                            num_training_steps = total_steps)
loss_fn = CrossEntropyLoss()

import os

models_dir = './models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

best_eval_accuracy = 0

for epoch_i in range(epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    total_train_loss = 0
    model.train()
    for step, (inputs, target) in enumerate(tqdm(train_dataloader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(**inputs).logits
        loss = loss_fn(output, target)
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    print("Average training loss: {0:.2f}".format(avg_train_loss))

    total_eval_accuracy = 0
    model.eval()
    with torch.no_grad():
        for step, (inputs, target) in enumerate(tqdm(val_dataloader)):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            optimizer.zero_grad()
            output = model(**inputs).logits

            acc, _ = flat_matrices(output, target)
            total_eval_accuracy += acc 

        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print("Average validation accuracy: {0:.2f}".format(avg_val_accuracy))
    if avg_val_accuracy > best_eval_accuracy:
        torch.save(model, './models/'+model_name)
        best_eval_accuracy = avg_val_accuracy
        
print("")
print("Training complete!")

"""## Test """

test_dataset = custom_dataset(test_df, feature)
test_dataloader = DataLoader(
            test_dataset,
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = test_batch_size # Evaluate with this batch size.
        )


model = torch.load('./models/'+model_name)
model.eval()

total_test_accuracy = 0
total_test_f1 = 0

with torch.no_grad():
    for step, (inputs, target) in enumerate(tqdm(test_dataloader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        
        y_preds = model(**inputs).logits

        acc, f1 = flat_matrices(y_preds, target)
        total_test_accuracy += acc 
        total_test_f1 += f1

avg_test_accuracy = total_test_accuracy / len(test_dataloader)
print("Accuracy: {0:.4f}".format(avg_test_accuracy))   

avg_test_f1 = total_test_f1 / len(test_dataloader)
print("F1: {0:.4f}".format(avg_test_f1))