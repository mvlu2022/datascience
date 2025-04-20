# Step 1: Import required libraries
import pandas as pd                           # For data handling
import seaborn as sns                         # To load Titanic dataset
import matplotlib.pyplot as plt               # For plotting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Step 2: Load Titanic dataset
df = sns.load_dataset('titanic')              # Load built-in Titanic dataset

# Step 3: Select important features and target variable
df = df[['survived', 'pclass', 'sex', 'age', 'fare']]  # Use minimal features

# Step 4: Drop missing values
df.dropna(inplace=True)                       # Remove rows with missing values

# Step 5: Convert categorical column 'sex' to numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)      # Convert 'sex' to 'sex_male' (1=male, 0=female)

# Step 6: Split features and target
X = df.drop('survived', axis=1)               # Features
y = df['survived']                            # Target (0 = died, 1 = survived)

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# ✅ Logistic Regression Model
# ------------------------------
log_model = LogisticRegression()              # Create logistic regression model
log_model.fit(X_train, y_train)               # Train the model

# Predict and evaluate
y_pred_log = log_model.predict(X_test)        # Predict on test set
print("🔹 Logistic Regression Results")
print("Accuracy :", accuracy_score(y_test, y_pred_log))         # Accuracy
print("Precision:", precision_score(y_test, y_pred_log))        # Precision
print("Recall   :", recall_score(y_test, y_pred_log))           # Recall
print(classification_report(y_test, y_pred_log))                # Full report

# ------------------------------
# ✅ Decision Tree Model
# ------------------------------
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limit depth for simplicity
tree_model.fit(X_train, y_train)               # Train the model

# Predict and evaluate
y_pred_tree = tree_model.predict(X_test)       # Predict on test set
print("🔹 Decision Tree Results")
print("Accuracy :", accuracy_score(y_test, y_pred_tree))        # Accuracy
print("Precision:", precision_score(y_test, y_pred_tree))       # Precision
print("Recall   :", recall_score(y_test, y_pred_tree))          # Recall
print(classification_report(y_test, y_pred_tree))               # Full report

# ------------------------------
# ✅ Plot the Decision Tree
# ------------------------------
plt.figure(figsize=(12, 6))                    # Set figure size
plot_tree(tree_model,                         # Plot tree
          feature_names=X.columns,            # Show feature names
          class_names=['Died', 'Survived'],   # Show class labels
          filled=True, rounded=True)          # Make it pretty
plt.title("🎯 Decision Tree (Depth=3)")
plt.show()
