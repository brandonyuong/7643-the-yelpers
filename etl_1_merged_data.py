import json
import pandas as pd
import numpy as np

# no truncation when printing df
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# download dataset from https://www.kaggle.com/yelp-dataset/yelp-dataset

#load reviews from downloaded json file
data_file = open("./data/yelp_academic_dataset_review.json", encoding="utf8")
data = []
for line in data_file:
    data.append(json.loads(line))
review_df = pd.DataFrame(data)
data_file.close()

# load business from downloaded json file
data_file = open("./data/yelp_academic_dataset_business.json", encoding="utf8")
data = []
for line in data_file:
    data.append(json.loads(line))
business_df = pd.DataFrame(data)
data_file.close()
data = None

review_df = review_df[['business_id', 'stars', 'text']]
# print(review_df.shape[0]) # 8635403

business_df = business_df[
    ['business_id', 'name', 'city', 'state', 'stars', 'review_count', 'categories']
]
business_df.columns = ['business_id', 'name', 'city', 'state', 'avg_stars',
                       'review_count', 'categories']
# print(business_df.shape[0]) # 160585

# review_df.dropna(inplace=True)
# print(review_df.shape[0]) # 8635403
business_df.dropna(subset=['categories'], inplace=True)
# print(business_df.shape[0]) # 160470

# print(business_df.head(3))

business_df = business_df[business_df['categories'].str.contains("estaurants")]
merged_df = pd.merge(business_df, review_df, on="business_id", how="left")


def get_sentiment(row):
    # For use with pandas apply().  Make sure row does not have NaN.
    if row['stars'] < 3:
        sentiment = -1
    elif row['stars'] > 3:
        sentiment = 1
    else:
        sentiment = 0
    return sentiment


merged_df['sentiment'] = merged_df.apply(get_sentiment, axis=1)
print(merged_df.head(10))
# print(merged_df.shape[0]) # 5574795

merged_df.to_csv('./data/yelp_merged_data.csv', index=False)
