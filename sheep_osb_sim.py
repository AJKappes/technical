import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# costs $/ewe for years 2010-15
vc = np.array([99.83, 108.19, 111.38, 113.18, 117.54, 115.67])
fc = np.array([14.03, 15.58, 16.37, 16.67, 17.44, 17.27])
cidr = 8

# 3-MA Q4 OSB holiday lamb price 20-90lbs live $/cwt
p = np.array([163.07, 207.31, 102.4, 186.94, 202.07, 179.56])
p_lb = p / 100

lamb_wt = np.arange(60, 91, .25)
flock = np.arange(0, 700, 1)

dist_df = pd.DataFrame(columns=['vc', 'fc', 'p', 'lamb_wt', 'flock'])
dist_list = [vc, fc, p, lamb_wt, flock]

for j in range(len(dist_list)):
    dist_df[dist_df.columns[j]] = stats.norm.rvs(np.mean(dist_list[j]), np.std(dist_list[j]), 1000)

for i in range(len(dist_df)):
    if dist_df.loc[i, 'flock'] < 0:
        dist_df.loc[i, 'flock'] = stats.norm.rvs(12.5, 3, 1)

df_sim = pd.DataFrame(columns=dist_df.columns)
for j in df_sim.columns:
    df_sim[j] = np.random.choice(dist_df[j], size=10000)

def profit(lamb_rate_reduction):
    return (2 * df_sim['p'] / 100 * df_sim['lamb_wt'] * (1 - lamb_rate_reduction) * df_sim['flock'] - \
           (df_sim['fc'] + df_sim['vc'] + cidr) * df_sim['flock']) / ((1 - lambing_red_rate[j]) * df_sim['flock'])

prof_cols = ['profit_25', 'profit_15', 'profit_10', 'profit_05', 'profit_0']
lambing_red_rate = [.25, .15, .1, .05, 0]
for j in range(len(prof_cols)):
    df_sim[prof_cols[j]] = profit(lambing_red_rate[j])

prof_out = np.array(df_sim[[j for j in df_sim.columns if 'profit' in j]].mean())
for i in range(len(prof_out)):
    print(round(prof_out[i], 2), 'mean profit $/lamb with',
          str(lambing_red_rate[i]) + '%', 'lambing rate reduction')
    print()

rate_reduction = np.linspace(0, 1, 200)
prof_rate_out = np.zeros(len(rate_reduction))
for i in range(len(rate_reduction)):
    prof_rate_out[i] = np.mean(profit(rate_reduction[i]))

be_lambing_rate = 1 - rate_reduction[np.argmin(abs(prof_rate_out))]
be_prof = prof_rate_out[np.argmin(abs(prof_rate_out))]

print('Break even lambing rate is', str(round(be_lambing_rate, 2)) + '%')
print()
print('$/ewe profits at break even lambing rate are', round(be_prof, 2))

# data table
df_rate_prof = pd.DataFrame(columns=['Lambing_Rate', 'Profit_Mean', 'Profit_Std'])
rates = np.array([0, .1, .2, .35, 1 - be_lambing_rate])
df_rate_prof['Lambing_Rate'] = 1 - rates
for i in range(len(df_rate_prof)):
    df_rate_prof.loc[i, 'Profit_Mean'] = np.mean(profit(rates[i]))
    df_rate_prof.loc[i, 'Profit_Std'] = np.std(profit(rates[i]))

print(round(df_rate_prof, 2).to_latex())
