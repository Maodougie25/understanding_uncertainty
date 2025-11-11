# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
from skimpy import skim

df = pd.read_csv('./data/nhanes_data_17_18.csv',low_memory=False)
print(df.head())
skim(df)

# %%

vars = ['CoveredByHealthInsurance',
        'HaveSeriousDifficultySeeing',
        'HaveSeriousDifficultyHearing',
        'WeightKg',
        'AlcoholGm_DR2TOT',
        'TakingInsulinNow',
         'AnnualHouseholdIncome']

race = pd.get_dummies(df['RacehispanicOriginWNhAsian'],dtype=int)#drop_first=True,dtype=int)
edu = pd.get_dummies(df['EducationLevelAdults20'],drop_first=True,dtype=int)
marital = pd.get_dummies(df['MaritalStatus'],drop_first=True,dtype=int)
gender = pd.get_dummies(df['Gender'],drop_first=True,dtype=int)

Z = df.loc[:,vars]
Z = pd.concat([race,Z,edu,marital,gender],axis=1)
Z = Z.dropna()
print(Z.shape)

Z.head()

# %%

y = Z['CoveredByHealthInsurance']
X = Z.drop('CoveredByHealthInsurance',axis=1)
#X = sm.add_constant(X) # I am including all the race dummies; no constant

reg = sm.GLM(y, X, family=sm.families.Binomial()).fit()
print(reg.summary())

# %%

y_hat = reg.predict(X)

sns.scatterplot(x=y,y=y_hat,alpha=.01) # Not well calibrated

# %%
