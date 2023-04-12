import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# dataset contains batches of weekly sold avocados in retail
df = pd.read_csv('figures/avocados_data/avocado.csv', sep=",")

# features we are considering are only:
# AveragePrice, Total Volume, PLU 4046, PLU 4225, PLU 4770
# the task is to classify based on this features the avocados into conventional
# and organic
df = df.drop(['region', 'year','XLarge Bags', 'Large Bags', 'Small Bags', 'Total Bags', 'Date', 'Unnamed: 0'], axis=1)

np.random.seed(7)
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
    df_subset = df.drop(np.random.choice(df.index, remove_n, replace=False))
    y = df_subset['type']
    X_normalized = df_subset.drop(['type'], axis=1)
    X_normalized = (X_normalized-X_normalized.mean())/X_normalized.std()
    df_subset_final = pd.concat([X_normalized, y], axis=1)
    
    # Visualize dataset
    sns.pairplot(df_subset_final, hue='type', kind='scatter', corner=True, plot_kws={'alpha': 0.5})
    plt.savefig(f"figures/avocados_data/data_rw_{key}", dpi=400)
    
    # seperate y data and transform y entries to 0 and 1
    y_num = []
    for row in y:
        if row == "organic":
            y_num.append(1)
        else:
            y_num.append(0)
    
    # Transform y and X to np arrays
    y = np.array(y_num)
    X = df_subset.drop(['type'], axis=1).to_numpy()

    np.save(f"X_rw_{key}", X)
    np.save(f"y_rw_{key}", y)