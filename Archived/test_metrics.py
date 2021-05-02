import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_seq_len = 250  # max input length

with open('test.npy', "rb") as f:
    X_test = np.load(f, allow_pickle=True)
    y_test = np.load(f, allow_pickle=True)

bert = AutoModel.from_pretrained('bert-base-cased')
MODEL_TYPE = 'bert-base-cased'
tokenizer = BertTokenizerFast.from_pretrained(MODEL_TYPE)

tokens_test = tokenizer.batch_encode_plus(
    X_test.tolist(),
    max_length=max_seq_len,
    padding='max_length',
    truncation=True,
    return_token_type_ids=False
)

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(y_test.tolist())


class BERT_Arch(nn.Module):
    """
    Model Architecture
    """
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)

        return x


model = BERT_Arch(bert)

# # push the model to GPU
model = model.to(device)


# ## Load Model

# load weights of best model
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))

# ## Test Model

# get predictions for test data
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

# model's performance
preds = np.argmax(preds, axis=1)
print(classification_report(test_y, preds))

# confusion matrix
pd.crosstab(test_y, preds)
