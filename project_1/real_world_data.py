import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# dataset contains batches of weekly sold avocados in retail
df = pd.read_csv('figures/avocados_data/avocado.csv', sep=",")

# remove non numerical or useless?? features
# features we are considering are only:
# AveragePrice, Total Volume, PLU 4046, PLU 4225, PLU 4770
# the task is to classify based on this features the avocados into conventional
# and organic
df = df.drop(['region', 'year','XLarge Bags', 'Large Bags', 'Small Bags', 'Total Bags', 'Date', 'Unnamed: 0'], axis=1)

# shuffle data set
df = df.sample(frac=1)

# 1,2,3,4 different sizes of datasets
# 1: whole dataset (18249 instances)
# 2: dataset with 15000 instances
# 3: dataset with 7500 instances
# 4: dataset with 1000 instances

removed_instances = {
    1: 0,
    2: df.shape[0] - 15000,
    3: df.shape[0] - 7500,
    4: df.shape[0] - 1000
}

for key, remove_n in removed_instances.items():
    drop_indices = np.random.choice(df.index, remove_n, replace=False)
    df_subset = df.drop(drop_indices)
    sns.pairplot(df_subset, hue='type')
    plt.savefig(f"data_{key}")
    y = df_subset['type']
    y_num = []
    for row in y:
        if row == "organic":
            y_num.append(1)
        else:
            y_num.append(0)

    y = np.array(y_num)
    X = df_subset.drop(['type'], axis=1).to_numpy()

    np.save(f"X_rw_{key}", X)
    np.save(f"y_rw_{key}", y)
    

