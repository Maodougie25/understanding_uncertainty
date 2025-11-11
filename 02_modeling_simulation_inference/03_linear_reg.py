# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
from skimpy import skim


## Load data, basic EDA:

df = pd.read_csv('./01_probability/data/ames_prices.csv',low_memory=False)
print(df.head())
skim(df)

# %%

## Visually examine outcome variable:

sns.kdeplot(df['price'])
plt.show()


# %% 

## Prepare data:

df['age'] = df['Yr.Sold']-df['Year.Built']
df['age_sq'] = df['age']**2

numeric_vars = ['price', 'area', 'Lot.Area', 'age', 'age_sq']
X_num = df.loc[:,numeric_vars]

# Create dummy variables/fixed effects/one hot encode:
ac = pd.get_dummies(df['Central.Air'],drop_first=True, dtype=int)
fireplaces = pd.get_dummies(df['Fireplaces'],drop_first=True, dtype=int)
type = pd.get_dummies(df['Bldg.Type'],drop_first=True, dtype=int)
style = pd.get_dummies(df['House.Style'],drop_first=True, dtype=int)
foundation = pd.get_dummies(df['Foundation'],drop_first=True, dtype=int)
quality = pd.get_dummies(df['Overall.Qual'],drop_first=True, dtype=int)

#all = pd.concat( [X_num, ac, fireplaces, type, style, foundation, quality], axis=1)
all = pd.concat( [X_num, ac, type, style, quality], axis=1)

print(all.shape)
all = all.dropna()
print(all.shape)

# %%

## Run linear regression:

y = all['price']
X = all.drop('price',axis=1)
X = sm.add_constant(X)

model = sm.OLS(y, X) # Linear regression object
reg = model.fit() # Fit model

print(reg.summary()) # Summary table

sns.scatterplot(x=y,y=reg.fittedvalues,alpha=.05).set(title='Actual versus Fitted')
plt.show()
