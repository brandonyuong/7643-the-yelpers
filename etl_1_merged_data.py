import json
import pandas as pd

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
# print(len(review_df)) # 8635403

business_df = business_df[
    ['business_id', 'name', 'city', 'state', 'stars', 'review_count', 'categories']
]
business_df.columns = ['business_id', 'name', 'city', 'state', 'avg_stars',
                       'review_count', 'categories']
# print(len(business_df)) #160470

review_df.dropna(subset=['stars'], inplace=True)
# print(len(review_df)) # 8635403
business_df.dropna(subset=['categories'], inplace=True)
# print(len(business_df)) # 160470

print(business_df.head(3))

business_df = business_df[business_df['categories'].str.contains("restaurants")]
merged_df = pd.merge(business_df, review_df, on="business_id", how="left")
merged_df.loc[(merged_df['stars'] < 3), 'sentiment'] = -1
merged_df.loc[(merged_df['stars'] > 3), 'sentiment'] = 1
print(merged_df.head(10))
