import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('titanic.csv')

df = df[['Age', 'Fare', 'Sex', 'Embarked']].dropna()

# Standardization (Z-score)
scaler_std = StandardScaler()
df[['Age_std', 'Fare_std']] = scaler_std.fit_transform(df[['Age', 'Fare']])

# Normalization (0 to 1)
scaler_norm = MinMaxScaler()
df[['Age_norm', 'Fare_norm']] = scaler_norm.fit_transform(df[['Age', 'Fare']])

df_dummies = pd.get_dummies(df, columns=['Sex', 'Embarked'])


print(df_dummies.head())
