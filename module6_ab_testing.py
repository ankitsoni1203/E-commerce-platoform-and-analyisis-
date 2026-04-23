import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

# Load Data
df = pd.read_csv("outputs/clean_ecommerce.csv", parse_dates=['order_purchase_timestamp'])
print(f"✅ Loaded: {df.shape}")

# ── Simulate A/B Test 
# Scenario: We sent a promotional email to group B (treatment)
# and nothing to group A (control). Did it increase order value?

customers = df['customer_unique_id'].unique()
n = len(customers)

# Randomly assign 50/50
control_ids   = np.random.choice(customers, size=n//2, replace=False)
treatment_ids = np.setdiff1d(customers, control_ids)

print(f"\n📊 Test Groups:")
print(f"   Control   (A): {len(control_ids):,} customers")
print(f"   Treatment (B): {len(treatment_ids):,} customers")

# Get their purchase behavior
control_df   = df[df['customer_unique_id'].isin(control_ids)]
treatment_df = df[df['customer_unique_id'].isin(treatment_ids)]

# Simulate treatment effect: promo email increases order value by ~8%
treatment_df = treatment_df.copy()
treatment_df['price'] = treatment_df['price'] * np.random.uniform(1.04, 1.12, len(treatment_df))

# Also simulate slightly higher conversion (treatment users order more)
# Add a few extra orders for treatment group
extra_orders = control_df.sample(int(len(control_df) * 0.04), random_state=7).copy()
extra_orders['customer_unique_id'] = np.random.choice(treatment_ids, len(extra_orders))
treatment_df = pd.concat([treatment_df, extra_orders], ignore_index=True)

# Metric 1: Conversion Rate ─
# Conversion = customer made at least one purchase
control_conversions   = control_df['customer_unique_id'].nunique()
treatment_conversions = treatment_df['customer_unique_id'].nunique()

control_conv_rate   = control_conversions   / len(control_ids)
treatment_conv_rate = treatment_conversions / len(treatment_ids)

print(f"\n📊 Conversion Rates:")
print(f"   Control   (A): {control_conv_rate*100:.2f}%")
print(f"   Treatment (B): {treatment_conv_rate*100:.2f}%")
print(f"   Lift          : {(treatment_conv_rate - control_conv_rate)*100:.2f}%")

# Chi-square test for conversion rate
converted_A   = control_conversions
not_conv_A    = len(control_ids) - converted_A
converted_B   = treatment_conversions
not_conv_B    = len(treatment_ids) - converted_B

contingency   = [[converted_A, not_conv_A],
                 [converted_B, not_conv_B]]
try:
    chi2, p_conv, dof, _ = chi2_contingency(contingency)
    print(f"\n   Chi-square test: chi2={chi2:.4f}, p={p_conv:.6f}")
    print(f"   Result: {'✅ Statistically significant' if p_conv < 0.05 else '❌ Not significant'} (p < 0.05)")
except ValueError:
    p_conv = 1.0  # No significance when test cannot be performed
    print(f"\n   Chi-square test: Cannot perform (both groups have 100% conversion)")
    print(f"   Result: ❌ Test not applicable")

# Metric 2: Average Order Value
aov_control   = control_df.groupby('order_id')['price'].sum()
aov_treatment = treatment_df.groupby('order_id')['price'].sum()

mean_aov_A = aov_control.mean()
mean_aov_B = aov_treatment.mean()

print(f"\n📊 Average Order Value:")
print(f"   Control   (A): BRL {mean_aov_A:.2f}")
print(f"   Treatment (B): BRL {mean_aov_B:.2f}")
print(f"   Lift          : {(mean_aov_B - mean_aov_A) / mean_aov_A * 100:.2f}%")

# t-test for AOV
t_stat, p_aov = ttest_ind(aov_control, aov_treatment, equal_var=False)
print(f"\n   Welch t-test: t={t_stat:.4f}, p={p_aov:.6f}")
print(f"   Result: {'✅ Statistically significant' if p_aov < 0.05 else '❌ Not significant'} (p < 0.05)")

# ── Metric 3: Revenue per Customer ─
rev_control   = control_df.groupby('customer_unique_id')['price'].sum().reindex(control_ids).fillna(0)
rev_treatment = treatment_df.groupby('customer_unique_id')['price'].sum().reindex(treatment_ids).fillna(0)

mean_rev_A = rev_control.mean()
mean_rev_B = rev_treatment.mean()

print(f"\n📊 Revenue per Customer:")
print(f"   Control   (A): BRL {mean_rev_A:.2f}")
print(f"   Treatment (B): BRL {mean_rev_B:.2f}")
print(f"   Lift          : {(mean_rev_B - mean_rev_A) / mean_rev_A * 100:.2f}%")

# Mann-Whitney U (non-parametric, revenue is skewed)
u_stat, p_rev = mannwhitneyu(rev_control, rev_treatment, alternative='less')
print(f"\n   Mann-Whitney U: U={u_stat:.0f}, p={p_rev:.6f}")
print(f"   Result: {'✅ Statistically significant' if p_rev < 0.05 else '❌ Not significant'} (p < 0.05)")

# Effect Size (Cohen's d) 
pooled_std = np.sqrt((rev_control.std()**2 + rev_treatment.std()**2) / 2)
cohens_d   = (mean_rev_B - mean_rev_A) / pooled_std
print(f"\n📊 Effect Size (Cohen's d): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:   effect = "Small"
elif abs(cohens_d) < 0.5: effect = "Medium"
else:                      effect = "Large"
print(f"   Interpretation: {effect} effect")

#  Power Analysis 
from scipy.stats import norm
alpha = 0.05
power = 0.80
z_alpha = norm.ppf(1 - alpha/2)
z_beta  = norm.ppf(power)
p1 = control_conv_rate
p2 = treatment_conv_rate
p_bar = (p1 + p2) / 2
required_n = int(2 * ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
                        z_beta  * np.sqrt(p1*(1-p1) + p2*(1-p2)))**2) /
               max((p2 - p1)**2, 1e-10))
print(f"\n📊 Power Analysis:")
print(f"   Required sample size per group: {required_n:,}")
print(f"   Actual size per group         : {n//2:,}")
print(f"   Test is {'✅ adequately powered' if n//2 >= required_n else '⚠️ underpowered'}")

#Final Summary
print("\n" + "="*55)
print("  A/B TEST SUMMARY REPORT")
print("="*55)
print(f"  Hypothesis  : Email promo increases revenue & conversion")
print(f"  Test Period : Simulated from actual customer data")
print(f"  Control n   : {len(control_ids):,}")
print(f"  Treatment n : {len(treatment_ids):,}")
print(f"")
print(f"  Conversion Lift : {(treatment_conv_rate - control_conv_rate)*100:.2f}%  (p={p_conv:.4f})")
print(f"  AOV Lift        : {(mean_aov_B - mean_aov_A)/mean_aov_A*100:.2f}%  (p={p_aov:.4f})")
print(f"  Revenue Lift    : {(mean_rev_B - mean_rev_A)/mean_rev_A*100:.2f}%  (p={p_rev:.4f})")
print(f"  Effect Size     : {cohens_d:.4f} ({effect})")
print(f"")
print(f"  Recommendation  : {'✅ LAUNCH the campaign' if p_conv < 0.05 and p_aov < 0.05 else '⚠️ More data needed'}")
print("="*55)

#Plots 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('A/B Test Analysis Results', fontsize=15, fontweight='bold')
COLORS = {'A (Control)': 'steelblue', 'B (Treatment)': 'coral'}

# 1. Conversion Rate
bars = axes[0,0].bar(['A (Control)','B (Treatment)'],
                     [control_conv_rate*100, treatment_conv_rate*100],
                     color=['steelblue','coral'])
axes[0,0].set_title(f'Conversion Rate\n(p={p_conv:.4f})')
axes[0,0].set_ylabel('%')
for bar in bars:
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11)
if p_conv < 0.05:
    axes[0,0].set_xlabel('✅ Statistically Significant', color='green', fontsize=10)

# 2. AOV Distribution
axes[0,1].hist(aov_control.clip(upper=aov_control.quantile(0.99)),
               bins=40, alpha=0.6, label='A (Control)',   color='steelblue', density=True)
axes[0,1].hist(aov_treatment.clip(upper=aov_treatment.quantile(0.99)),
               bins=40, alpha=0.6, label='B (Treatment)', color='coral',     density=True)
axes[0,1].axvline(mean_aov_A, color='steelblue', linestyle='--', linewidth=2)
axes[0,1].axvline(mean_aov_B, color='coral',     linestyle='--', linewidth=2)
axes[0,1].set_title(f'Order Value Distribution\n(p={p_aov:.4f})')
axes[0,1].set_xlabel('Order Value (BRL)')
axes[0,1].legend()

# 3. Revenue per customer box
rev_data  = pd.DataFrame({'group': ['A']*len(rev_control) + ['B']*len(rev_treatment),
                           'revenue': list(rev_control) + list(rev_treatment)})
rev_clip  = rev_data[rev_data['revenue'] <= rev_data['revenue'].quantile(0.98)]
sns.boxplot(data=rev_clip, x='group', y='revenue', ax=axes[0,2],
            palette={'A':'steelblue','B':'coral'})
axes[0,2].set_title(f'Revenue per Customer\n(p={p_rev:.4f})')
axes[0,2].set_ylabel('Revenue (BRL)')

# 4. Metrics comparison
metrics = ['Conversion\nLift %', 'AOV\nLift %', 'Revenue\nLift %']
lifts   = [
    (treatment_conv_rate - control_conv_rate) / control_conv_rate * 100,
    (mean_aov_B - mean_aov_A) / mean_aov_A * 100,
    (mean_rev_B - mean_rev_A) / mean_rev_A * 100
]
colors  = ['green' if l > 0 else 'red' for l in lifts]
bars2   = axes[1,0].bar(metrics, lifts, color=colors, alpha=0.75)
axes[1,0].axhline(0, color='black', linewidth=0.8)
axes[1,0].set_title('Lift % — Treatment vs Control')
axes[1,0].set_ylabel('Lift %')
for bar, val in zip(bars2, lifts):
    axes[1,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.4,
                   f'{val:+.1f}%', ha='center', va='bottom', fontsize=11)

# 5. P-values summary
pvals  = [p_conv, p_aov, p_rev]
labels = ['Conversion\n(Chi-sq)', 'AOV\n(t-test)', 'Revenue\n(Mann-Whitney)']
bar_colors = ['green' if p < 0.05 else 'red' for p in pvals]
axes[1,1].bar(labels, pvals, color=bar_colors, alpha=0.75)
axes[1,1].axhline(0.05, color='black', linestyle='--', linewidth=1.5, label='α = 0.05')
axes[1,1].set_title('P-values per Metric')
axes[1,1].set_ylabel('p-value')
axes[1,1].legend()
for i, (bar, pv) in enumerate(zip(axes[1,1].patches, pvals)):
    axes[1,1].text(i, pv + 0.002, f'{pv:.4f}', ha='center', va='bottom', fontsize=10)

# 6. Cumulative revenue over time (illustrative)
n_days = 90
days   = np.arange(1, n_days + 1)
cum_A  = np.cumsum(np.random.normal(mean_rev_A / 30, 5, n_days))
cum_B  = np.cumsum(np.random.normal(mean_rev_B / 30, 5, n_days))
axes[1,2].plot(days, cum_A, label='A (Control)',   color='steelblue', linewidth=2)
axes[1,2].plot(days, cum_B, label='B (Treatment)', color='coral',     linewidth=2)
axes[1,2].fill_between(days, cum_A, cum_B, alpha=0.15, color='green')
axes[1,2].set_title('Simulated Cumulative Revenue (90 days)')
axes[1,2].set_xlabel('Day')
axes[1,2].set_ylabel('Cumulative Revenue (BRL)')
axes[1,2].legend()

plt.tight_layout()
plt.savefig("outputs/module6_ab_test.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n Module 6 Complete!")
print("   Plot saved: outputs/module6_ab_test.png")
