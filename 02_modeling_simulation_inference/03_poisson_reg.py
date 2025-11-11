# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from skimpy import skim
from statsmodels.discrete.discrete_model import Poisson

df = pd.read_csv('./01_probability/data/metabric.csv',low_memory=False)
print(df.head())
skim(df)

# %%

## Clean data:

num_vars = [
    'Tumor Size', 
    'Lymph nodes examined positive',
    'Age at Diagnosis',
    'Mutation Count',
    'Nottingham prognostic index',
    'TMB (nonsynonymous)',
    'Tumor Stage',]

all = df.loc[:,num_vars]
all = all.dropna()

X = all.drop(['Tumor Size', 'Lymph nodes examined positive'],axis=1)
X = sm.add_constant(X)

# %%

## Regressions on Positive Lymph Nodes

y = all['Lymph nodes examined positive']
model = Poisson(y, X)
reg = model.fit()
print(reg.summary())

y_hat = reg.predict(X)
sns.histplot(y_hat,bins=50)
plt.show()
sns.histplot(y, bins=50)
plt.show()


# %%

## Regressions on Tumor Size

y = all['Tumor Size']
model = Poisson(y, X)
reg = model.fit()
print(reg.summary())

y_hat = reg.predict(X)
sns.histplot(y_hat,bins=50)
plt.show()
sns.histplot(y, bins=50)
plt.show()


# %%
