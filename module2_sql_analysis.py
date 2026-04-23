import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs", exist_ok=True)

# Load Clean Data 
df = pd.read_csv("outputs/clean_ecommerce.csv", parse_dates=['order_purchase_timestamp'])
print(f" Loaded clean data: {df.shape}")

# Load into SQLite
conn = sqlite3.connect("outputs/ecommerce.db")
df.to_sql("orders", conn, if_exists="replace", index=False)
print(" Data loaded into SQLite database")

# SQL Query Helper
def run_query(sql):
    return pd.read_sql_query(sql, conn)


# RFM SEGMENTATION
rfm_sql = """
SELECT
    customer_unique_id,
    CAST(JULIANDAY('2018-10-01') - JULIANDAY(MAX(order_purchase_timestamp)) AS INTEGER) AS recency,
    COUNT(DISTINCT order_id)  AS frequency,
    ROUND(SUM(price), 2)      AS monetary
FROM orders
WHERE order_purchase_timestamp < '2018-10-01'
GROUP BY customer_unique_id
"""
rfm = run_query(rfm_sql)
print(f"\n RFM table created: {rfm.shape}")
print(rfm.describe().round(2))

# Score each dimension 1-5
rfm['R_score'] = pd.qcut(rfm['recency'],5,labels=[5,4,3,2,1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M_score'] = pd.qcut(rfm['monetary'],5,labels=[1,2,3,4,5]).astype(int)
rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# Segment labels
def segment(score):
    if score >= 13: return 'Champions'
    elif score >= 10: return 'Loyal Customers'
    elif score >= 7: return 'Potential Loyalists'
    elif score >= 5: return 'At Risk'
    else: return 'Lost'

rfm['Segment'] = rfm['RFM_score'].apply(segment)
rfm.to_csv("outputs/rfm_segments.csv", index=False)
print("\n📊 RFM Segments:")
print(rfm['Segment'].value_counts())

# performing  COHORT RETENTION ANALYSIS

cohort_sql = """
SELECT
    customer_unique_id,
    order_id,
    order_purchase_timestamp
FROM orders
"""
cohort_df = run_query(cohort_sql)
cohort_df['order_purchase_timestamp'] = pd.to_datetime(cohort_df['order_purchase_timestamp'])
cohort_df['order_month'] = cohort_df['order_purchase_timestamp'].dt.to_period('M')

# First purchase month = cohort
cohort_df['cohort'] = cohort_df.groupby('customer_unique_id')['order_month'].transform('min')
cohort_df['period_number'] = (cohort_df['order_month'] - cohort_df['cohort']).apply(lambda x: x.n)

cohort_data = cohort_df.groupby(['cohort','period_number'])['customer_unique_id'].nunique().reset_index()
cohort_data.columns = ['cohort','period_number','customers']

cohort_pivot = cohort_data.pivot_table(index='cohort', columns='period_number', values='customers')
cohort_size  = cohort_pivot.iloc[:, 0]
retention    = cohort_pivot.divide(cohort_size, axis=0).round(3)*100
# Keep only first 6 periods for readability
retention = retention.iloc[:,:6]
retention.index = retention.index.astype(str)
print("\n Cohort retention table created")
print(retention.head().round(1))

# Additional SQL Queries 
# Revenue by state
state_rev = run_query("""
SELECT customer_state, ROUND(SUM(price),2) AS revenue, COUNT(DISTINCT order_id) AS orders
FROM orders GROUP BY customer_state ORDER BY revenue DESC LIMIT 10
""")

# Payment method breakdown
payment = run_query("""
SELECT payment_type, COUNT(*) AS count, ROUND(SUM(payment_value),2) AS total_value
FROM orders GROUP BY payment_type ORDER BY count DESC
""")

print("\n Top States by Revenue:")
print(state_rev.to_string(index=False))
print("\n Payment Methods:")
print(payment.to_string(index=False))

conn.close()

# Plots 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('SQL Analysis Results', fontsize=15, fontweight='bold')

# 1. RFM Segment Distribution
seg_counts = rfm['Segment'].value_counts()
colors = ['#2ecc71','#3498db','#f39c12','#e74c3c','#95a5a6']
axes[0].pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%', colors=colors, startangle=140)
axes[0].set_title('Customer Segments (RFM)')

# 2 Cohort Retention Heatmap
sns.heatmap(
    retention.fillna(0).astype(float),
    annot=True, fmt='.0f', cmap='YlGnBu',
    ax=axes[1], linewidths=0.5, cbar_kws={'label': 'Retention %'}
)

axes[1].set_title('Cohort Retention % (first 6 months)')
axes[1].set_xlabel('Month Number')
axes[1].set_ylabel('Cohort')
axes[1].tick_params(axis='y', rotation=0)



# 3. Revenue by State


axes[2].barh(state_rev['customer_state'][::-1], state_rev['revenue'][::-1], color='steelblue')
axes[2].set_title('Top 10 States by Revenue')
axes[2].set_xlabel('Revenue (BRL)')
plt.tight_layout()
plt.savefig("outputs/module2_sql.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n Module 2 Complete! Plot saved: outputs/module2_sql.png")
  
