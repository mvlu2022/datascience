import seaborn as sns
import pandas as pd
import statsmodels.api as sm

titanic = sns.load_dataset('titanic')

simple_data = titanic.dropna(subset=['age', 'fare'])
X_simple = sm.add_constant(simple_data['fare'])   # add intercept
y_simple = simple_data['age']
simple_model = sm.OLS(y_simple, X_simple).fit()

print("=== Simple Linear Regression: Age ~ Fare ===")
print(simple_model.summary())

titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

multi_data = titanic.dropna(subset=['age', 'fare', 'pclass', 'sex'])
X_multi = sm.add_constant(multi_data[['fare', 'pclass', 'sex']])  # Add intercept
y_multi = multi_data['age']
multi_model = sm.OLS(y_multi, X_multi).fit()

print("\nMultiple Linear Regression: Age ~ Fare + Pclass + Sex")
print(multi_model.summary())
