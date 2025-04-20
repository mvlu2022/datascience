import seaborn as sns
import pandas as pd
import statsmodels.api as sm

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# ========== SIMPLE LINEAR REGRESSION ==========
# Predict 'age' from 'fare'
simple_data = titanic.dropna(subset=['age', 'fare'])
X_simple = sm.add_constant(simple_data['fare'])   # Add intercept
y_simple = simple_data['age']
simple_model = sm.OLS(y_simple, X_simple).fit()

print("=== Simple Linear Regression: Age ~ Fare ===")
print(simple_model.summary())

# ========== MULTIPLE LINEAR REGRESSION ==========
# Convert 'sex' to numeric: male=0, female=1
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

# Drop rows with missing values in predictors
multi_data = titanic.dropna(subset=['age', 'fare', 'pclass', 'sex'])
X_multi = sm.add_constant(multi_data[['fare', 'pclass', 'sex']])  # Add intercept
y_multi = multi_data['age']
multi_model = sm.OLS(y_multi, X_multi).fit()

print("\n=== Multiple Linear Regression: Age ~ Fare + Pclass + Sex ===")
print(multi_model.summary())
