import pandas as pd

df = pd.read_csv('titanic.csv')
print('before handling Age column:')
print(df.isnull().sum())

df['Age'].fillna(df['Age'].mean(), inplace=True)
df.dropna(subset=['Embarked'], inplace=True)

print('after handling Age column:')
print(df.isnull().sum())

# removing outlier from Fare column
Q1, Q3 = df['Fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

df_filtered = df[df['Age'] > 30]

df_sorted = df.sort_values(by='Fare', ascending=False)

# Group: average age by Pclass
df_grouped = df.groupby('Pclass')['Age'].mean()

print(df_filtered.head())
print(df_sorted[['Name', 'Fare']].head())
print(df_grouped)
