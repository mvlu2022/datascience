import seaborn as sns

from scipy.stats import f_oneway

from statsmodels.stats.multicomp import pairwise_tukeyhsd

titanic = sns.load_dataset('titanic').dropna(subset=['age', 'pclass'])

group1 = titanic[titanic['pclass'] == 1]['age']
group2 = titanic[titanic['pclass'] == 2]['age']
group3 = titanic[titanic['pclass'] == 3]['age']

f_stat, p_value = f_oneway(group1, group2, group3)

if p_value < 0.05:
    print("Reject Null Hypothesis: At least one class mean age is different.")
else:
    print("Fail to Reject Null Hypothesis: No significant difference in mean ages.")

tukey = pairwise_tukeyhsd(endog=titanic['age'], groups=titanic['pclass'], alpha=0.05)
print("\nPost-hoc Test Result:\n", tukey)

print("\nMean Ages by Class:")
print(titanic.groupby('pclass')['age'].mean())
