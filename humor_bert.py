import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast

# specify device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('./data/yelp_reduced.csv')

X_train, X_test, y_train, y_test = train_test_split(df.text,
                                                    df.humor,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=df.humor)

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-cased')

MODEL_TYPE = 'bert-base-cased'

tokenizer = BertTokenizerFast.from_pretrained(MODEL_TYPE)
