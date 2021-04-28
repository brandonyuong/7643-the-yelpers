#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:


# %cd "/content/drive/MyDrive/cs7643-deep-learning/project"


# In[3]:


# !nvidia-smi -L


# In[4]:


# !pip install -q -U watermark


# In[5]:


# !pip install -qq transformers


# In[6]:


import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import timeit
import gc
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 77
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# In[7]:


# data = pd.read_csv("yelp_humor_merged_v1.csv")
# data = pd.read_csv("./data/yelp_merged_data.csv")
# data


# In[8]:


# word_cnt_threshold=400
# humor_threshold=10


# In[9]:


# df = data[(data['word_count'] <= word_cnt_threshold) & ((data['funny']>=humor_threshold) | (data['funny'] == 0))]
# df


# In[10]:


# df['humor'].value_counts()


# In[11]:


# df['funny'].value_counts()


# In[12]:


# humor_sample = min(10000, df['humor'].value_counts()[1])
# humor_sample


# In[13]:


# not_humor_split = 0.5
# not_humor_sample = int(humor_sample * (not_humor_split) / (1 - not_humor_split))
# not_humor_sample


# In[14]:


# df_humor = df[df['humor']==1]
# df_humor = df_humor.sample(n=humor_sample, random_state=RANDOM_SEED)
# df_humor


# In[15]:


# df_not_humor = df[df['humor']==0]
# df_not_humor = df_not_humor.sample(n=not_humor_sample, random_state=RANDOM_SEED)
# df_not_humor


# In[16]:


# df = pd.concat([df_humor, df_not_humor])
df = pd.read_csv("data/yelp_reduced_20k.csv")
print(df)

# In[17]:


class_names = ['not funny', 'funny']

# In[18]:


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
# PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'


# In[19]:


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# In[20]:


# token_lens = []
#
# for txt in df.text:
#     tokens = tokenizer.encode(txt, truncation=True)
#     token_lens.append(len(tokens))

# In[21]:


# sns.displot(token_lens)
# plt.xlim([0, 512])
# plt.xlabel('Token count')

# In[22]:


# (np.array(token_lens) == 512).sum()

# In[23]:


# del data, df_humor, df_not_humor, tokens
# del tokens
gc.collect()

# In[24]:


MAX_LEN = 512


# In[25]:


class GPReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


# In[26]:


df_train, df_test = train_test_split(df, test_size=0.3, stratify=df.humor,
                                     random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, stratify=df_test.humor,
                                   random_state=RANDOM_SEED)

# In[27]:


df_train['humor'].value_counts()

# In[28]:


df_val['humor'].value_counts()

# In[29]:


df_test['humor'].value_counts()

# In[30]:


print(df_train.shape, df_val.shape, df_test.shape)


# In[31]:


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.text.to_numpy(),
        targets=df.humor.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2
    )


# In[32]:


BATCH_SIZE = 8

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# In[ ]:


data = next(iter(train_data_loader))
print(data.keys())

# In[ ]:


print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)


# In[ ]:


class HumorClassifier(nn.Module):

    def __init__(self, n_classes):
        super(HumorClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


# In[ ]:


model = HumorClassifier(len(class_names))
model = model.to(device)

# In[ ]:


input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape)  # batch size x seq length
print(attention_mask.shape)  # batch size x seq length

# In[ ]:


F.softmax(model(input_ids, attention_mask), dim=1)

# In[ ]:


EPOCHS = 4

optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


# In[ ]:


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


# In[ ]:


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


# In[ ]:


history = defaultdict(list)
best_accuracy = 0

start = timeit.default_timer()
for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

stop = timeit.default_timer()
train_time = (stop - start) / 60
print("\nTrain Time: " + str(train_time) + " minutes\n")

# In[ ]:


plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
# In[ ]:


model = HumorClassifier(len(class_names))
model.load_state_dict(torch.load('best_model_state.bin'))
model = model.to(device)

# In[ ]:


test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
)

test_acc.item()


# In[ ]:


def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


# In[ ]:


y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    test_data_loader
)

# In[ ]:


print(classification_report(y_test, y_pred, target_names=class_names))


# In[ ]:


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True humor')
    plt.xlabel('Predicted humor')


cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

# In[ ]:


idx = 2

review_text = y_review_texts[idx]
true_sentiment = y_test[idx]
pred_df = pd.DataFrame({
    'class_names': class_names,
    'values': y_pred_probs[idx]
})

# In[ ]:


print("\n".join(wrap(review_text)))
print()
print(f'True humor: {class_names[true_sentiment]}')

# In[ ]:


sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
plt.ylabel('humor')
plt.xlabel('probability')
plt.xlim([0, 1])

# In[ ]:
