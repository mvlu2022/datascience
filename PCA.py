import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
df=sns.load_dataset('titanic')[['age','fare','sex','pclass','survived']].dropna()
df['sex']=LabelEncoder().fit_transform(df['sex'])
x=df[['age','fare','sex','pclass']]
y=df['survived']
x_scaled=StandardScaler().fit_transform(x)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x_scaled)
print("PCA Components: \n",pca.components_)
print("Explained Varience Ratio: ",pca.explained_variance_ratio_)

plt.scatter(x_pca[:,0],x_pca[:,1], c=y,cmap='coolwarm')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA on Titanic dataset')
plt.show()