
import pandas as pd                           
import seaborn as sns                         
import matplotlib.pyplot as plt               
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


df = sns.load_dataset('titanic')


df = df[['survived', 'pclass', 'sex', 'age', 'fare']]

df.dropna(inplace=True)


df = pd.get_dummies(df, drop_first=True)      
X = df.drop('survived', axis=1)               
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression()              
log_model.fit(X_train, y_train)               

y_pred_log = log_model.predict(X_test)        
print("Logistic Regression Results")
print("Accuracy :", accuracy_score(y_test, y_pred_log))         
print("Precision:", precision_score(y_test, y_pred_log))        
print("Recall   :", recall_score(y_test, y_pred_log))           
print(classification_report(y_test, y_pred_log))                

tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)  
tree_model.fit(X_train, y_train)               

y_pred_tree = tree_model.predict(X_test)       
print("Decision Tree Results")
print("Accuracy :", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree))       
print("Recall   :", recall_score(y_test, y_pred_tree))          
print(classification_report(y_test, y_pred_tree))               


plt.figure(figsize=(12, 6))                    
plot_tree(tree_model,                         
          feature_names=X.columns,            
          class_names=['Died', 'Survived'],   
          filled=True, rounded=True)          
plt.title("Decision Tree (Depth=3)")
plt.show()
