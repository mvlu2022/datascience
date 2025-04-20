# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Select relevant columns
df = df[['fare', 'age', 'pclass', 'sex', 'embarked']]

# Drop missing values
df.dropna(inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop('fare', axis=1)
y = df['fare']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Coefficient analysis
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\n Model Coefficients:")
print(coeff_df)

# Intercept
print(f"\n Intercept: {model.intercept_:.2f}")

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color='orange')
plt.title('Residuals Distribution')
plt.xlabel('Residuals (Actual Fare - Predicted Fare)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
