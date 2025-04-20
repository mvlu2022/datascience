
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Use only a few columns and drop rows with missing values
df = df[['survived', 'age', 'fare', 'sex']].dropna()

# Convert 'sex' to numeric: male=1, female=0
df['sex'] = df['sex'].map({'male': 1, 'female': 0})

# Features and target
X = df[['age', 'fare', 'sex']]
y = df['survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train the Decision Tree model
model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
print("Model Evaluation Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred))

# Plot the decision tree
plt.figure(figsize=(8, 6))
plot_tree(model, feature_names=X.columns, class_names=['Died', 'Survived'], filled=True)
plt.title("Simple Decision Tree (Titanic)")
plt.show()
