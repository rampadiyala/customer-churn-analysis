"""Churn analysis on randomly generated sample data. Results are not real business findings."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rng = np.random.default_rng(1)
n = 1000
df = pd.DataFrame({'contract': rng.choice(['Monthly','Yearly'], n), 'tenure': rng.integers(1,61,n).astype(float)})
chance = np.where(df.contract=='Monthly', .35, .12) + np.where(df.tenure<12, .15, 0)
df['churn'] = (rng.random(n) < chance).astype(int)
df.loc[rng.choice(n, 30, replace=False), 'tenure'] = np.nan  # simulate missing values
df['tenure'] = df.tenure.fillna(df.tenure.median())
df['tenure_group'] = pd.cut(df.tenure, [0,12,36,60], labels=['0-12','13-36','37-60'])
for col in ['contract','tenure_group']:
    rate = df.groupby(col, observed=True).churn.mean()*100
    print(rate.round(1))
    ax = rate.plot.bar(title=f'Churn rate by {col} (sample data)', ylabel='%')
    ax.figure.tight_layout(); ax.figure.savefig(f'churn_by_{col}.png'); plt.close()
