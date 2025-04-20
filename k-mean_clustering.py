import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
df=sns.load_dataset('titanic')[['age','fare','sex','survived']].dropna()
df['sex']=LabelEncoder().fit_transform(df['sex'])
x=df[['sex','age','fare']]
y=df['survived']
kmeans=KMeans(n_clusters=2, random_state=5)
df['cluster']=kmeans.fit_predict(x)

print("Number of Cluster: ",len(np.unique(df['cluster'])))
print("Predicted Cluster Label: ",df['cluster'].values[:10])
print("Actual Survived Value: ",y.values[:10])

matching=(df['cluster']==y).sum()
accuracy=matching/len(df)

print("Rough Accuracy (Cluster vs Survived):", round(accuracy * 100, 2), "%")
plt.scatter(df['age'],df['fare'],c=df['cluster'],cmap='coolwarm')
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("K-Mean Clustering Ploting Graphs")
plt.show()