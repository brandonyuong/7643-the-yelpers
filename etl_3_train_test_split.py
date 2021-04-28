import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

result = pd.read_csv("data/yelp_reduced_10k.csv")
print(result.shape)
X_train, X_temp, y_train, y_temp = train_test_split(result.text,
                                                    result.humor,
                                                    test_size=0.3,
                                                    random_state=8,
                                                    stratify=result.humor)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                random_state=8,
                                                test_size=0.5,
                                                stratify=y_temp)

with open('train.npy', "wb") as f:
    np.save(f, X_train)
    np.save(f, y_train)

with open('val.npy', "wb") as f:
    np.save(f, X_val)
    np.save(f, y_val)

with open('test.npy', "wb") as f:
    np.save(f, X_test)
    np.save(f, y_test)
