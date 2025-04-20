import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')

# Drop rows with missing age
df = df.dropna(subset=['age'])

# Split data
survivors = df[df['survived'] == 1]['age']
non_survivors = df[df['survived'] == 0]['age']

# Perform t-test
t_stat, p_value = ttest_ind(survivors, non_survivors)

# Print result
alpha = 0.05
if p_value < alpha:
    print("Reject Null Hypothesis: Mean age differs between survivors and non-survivors.")
else:
    print("Fail to Reject Null Hypothesis: No significant difference in mean age.")

# Plot age distribution
plt.figure(figsize=(8, 5))
sns.histplot(survivors, color='green', label='Survived', kde=True, stat="density", bins=20)
sns.histplot(non_survivors, color='red', label='Not Survived', kde=True, stat="density", bins=20)
plt.title('Age Distribution: Survivors vs Non-Survivors')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()
