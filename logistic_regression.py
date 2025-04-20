import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
df=sns.load_dataset("titanic").dropna(subset=['sex','age','survived'])
df['sex']=LabelEncoder().fit_transform(df['sex'])
x=df[['sex','age']]
y=df['survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LogisticRegression()
model.fit(x_train,y_train)
predc=model.predict(x_test)
print("Accuracy: ",accuracy_score(y_test,predc))
print("Predicted value",predc[:10])
print("Actual value",y_test[:10])
plt.scatter(x_test['age'],x_test['sex'], c=predc, cmap='bwr')
plt.xlabel('Age')
plt.ylabel('sex(male=1 and female=0)')
plt.title('Logistic Regression Ploting')
plt.show()  