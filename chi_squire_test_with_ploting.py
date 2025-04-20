import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Drop missing values in 'sex' or 'survived'
titanic = titanic.dropna(subset=['sex', 'survived'])

# Hypotheses:
# H0: Survival is independent of gender
# H1: Survival depends on gender

# Create a contingency table
table = pd.crosstab(titanic['sex'], titanic['survived'])

# Chi-square test
chi2, p, dof, expected = chi2_contingency(table)

# Print the result
alpha = 0.05
if p < alpha:
    print("Reject Null Hypothesis: Survival depends on gender.")
else:
    print("Fail to Reject Null Hypothesis: Survival does not depend on gender.")

# Plot the results using countplot
plt.figure(figsize=(8, 5))
sns.countplot(data=titanic, x='sex', hue='survived', palette=['red', 'green'])
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Passenger Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()
