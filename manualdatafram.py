import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
student_data=[(1,"Abhishek",20),(2,"umang",20),(3,"Abhi"),(4,"Ritesh",15),(5,"Rajit",12)]
df=pd.DataFrame(student_data,columns=['id','name','age'])
#print(df)
# print("*"*30)
# print(df.columns)
# print(df.tail(3))
# print(df.head(2))
# print(df.size)
# print(df.dtypes)
# print(df.shape)
# print("corrwith")
# print(df.corrwith)
# print('checking for missing values')
# print(df.isnull().sum())

# print(df.isnull().sum()[df.isnull().sum()>0])
# sns.heatmap(df.isnull(),cbar=False,cmap="viridis",yticklabels=False)
# plt.title("Visualize of missing data")
# plt.show()

# print(df.dropna())
# print(df.fillna(0))
# print(df['age'].fillna(1))
# print(df.fillna(method='ffill'))# forward fill
# print(df.fillna(method='bfill'))# backword fill
# print(df['age'].fillna(df['age'].mean()))
# print(df['age'].fillna(df['age'].median()))
# print(df['age'].fillna(df['age'].mode()[0]))
# df['age']=df['age'].interpolate()
# print(df['age'])

print(df.rename(columns={'id':'n_id'},inplace=True))
print(df.columns)

df['n_id']=df['n_id'].astype(str)
print(df.dtypes)