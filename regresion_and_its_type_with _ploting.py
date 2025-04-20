import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# ========== SIMPLE LINEAR REGRESSION ==========
simple_data = titanic.dropna(subset=['age', 'fare'])
X_simple = sm.add_constant(simple_data['fare'])
y_simple = simple_data['age']
simple_model = sm.OLS(y_simple, X_simple).fit()

print("=== Simple Linear Regression: Age ~ Fare ===")
print(simple_model.summary())

# 🔹 Plot Simple Linear Regression
plt.figure(figsize=(7, 5))
sns.scatterplot(x='fare', y='age', data=simple_data, label='Data')
pred_y = simple_model.predict(X_simple)
plt.plot(simple_data['fare'], pred_y, color='red', label='Regression Line')
plt.title('Simple Linear Regression: Age ~ Fare')
plt.xlabel('Fare')
plt.ylabel('Age')
plt.legend()
plt.tight_layout()
plt.show()

# ========== MULTIPLE LINEAR REGRESSION ==========
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
multi_data = titanic.dropna(subset=['age', 'fare', 'pclass', 'sex'])
X_multi = sm.add_constant(multi_data[['fare', 'pclass', 'sex']])
y_multi = multi_data['age']
multi_model = sm.OLS(y_multi, X_multi).fit()

print("\n=== Multiple Linear Regression: Age ~ Fare + Pclass + Sex ===")
print(multi_model.summary())

# Plot Coefficients of Multiple Regression
coeffs = multi_model.params.drop('const')  # Exclude intercept
plt.figure(figsize=(6, 4))
coeffs.plot(kind='bar', color='skyblue')
plt.title('Multiple Regression Coefficients')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
