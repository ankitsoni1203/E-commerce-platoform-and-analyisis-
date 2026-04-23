import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = r"C:\e-commerce\archive"
os.makedirs("outputs", exist_ok=True)

orders = pd.read_csv(os.path.join(DATA_PATH, "olist_orders_dataset.csv"))
order_items = pd.read_csv(os.path.join(DATA_PATH, "olist_order_items_dataset.csv"))
order_payments = pd.read_csv(os.path.join(DATA_PATH, "olist_order_payments_dataset.csv"))
order_reviews = pd.read_csv(os.path.join(DATA_PATH, "olist_order_reviews_dataset.csv"))
customers = pd.read_csv(os.path.join(DATA_PATH, "olist_customers_dataset.csv"))
products = pd.read_csv(os.path.join(DATA_PATH, "olist_products_dataset.csv"))
category_trans = pd.read_csv(os.path.join(DATA_PATH, "product_category_name_translation.csv"))

print("All datasets loaded")

print(f" Orders: {orders.shape} | Items: {order_items.shape} | Customers: {customers.shape}")
df = orders.merge(order_items, on="order_id", how="left")
df = df.merge(order_payments, on="order_id", how="left")
df = df.merge(customers, on="customer_id", how="left")
df = df.merge(products, on="product_id", how="left")
df = df.merge(category_trans, on="product_category_name", how="left")

print(f"\n Merged shape: {df.shape}")
date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n Duplicate removed: {before - len(df)}")
null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print("\n📊 Null % per column (only non-zero):")
print(null_pct[null_pct > 0].round(2).to_string())

# drop nulls which are important----
df.dropna(subset=['customer_id', 'order_id', 'price'], inplace=True)

# Fills Non-Critical Nulls--
df['product_category_name_english'].fillna('unknown', inplace=True)
df['order_approved_at'].fillna(df['order_purchase_timestamp'], inplace=True)

# Keep Only Delivered Orders
df = df[df['order_status'] == 'delivered'].copy()

# Remove Price Outliers 
price_cap = df['price'].quantile(0.995)
df = df[df['price'] <= price_cap].copy()

# Feature Engineering 
df['year'] = df['order_purchase_timestamp'].dt.year
df['month'] = df['order_purchase_timestamp'].dt.month
df['day_of_week'] = df['order_purchase_timestamp'].dt.day_name()
df['hour'] = df['order_purchase_timestamp'].dt.hour
df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

print(f"\n Clean dataset shape: {df.shape}")

# Saving Clean Data 
df.to_csv("outputs/clean_ecommerce.csv", index=False)
print(" Saved: outputs/clean_ecommerce.csv")

# EDA Ploting  
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(3, 2, figsize=(15, 14))
fig.suptitle('E-Commerce Exploratory Data Analysis', fontsize=17, fontweight='bold', y=1.01)

# 1. Monthly Revenue
monthly_rev = df.groupby(df['order_purchase_timestamp'].dt.to_period('M'))['price'].sum()
monthly_rev.index = monthly_rev.index.astype(str)
axes[0,0].plot(monthly_rev.index, monthly_rev.values, marker='o', color='steelblue', linewidth=2)
axes[0,0].set_title('Monthly Revenue (BRL)')
axes[0,0].set_xlabel('Month')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].set_ylabel('Revenue')

# 2. Top 10 Categories
top_cat = df.groupby('product_category_name_english')['price'].sum().nlargest(10)
axes[0,1].barh(top_cat.index[::-1], top_cat.values[::-1], color='coral')
axes[0,1].set_title('Top 10 Categories by Revenue')
axes[0,1].set_xlabel('Revenue (BRL)')

# 3. Price Distribution
axes[1,0].hist(df['price'], bins=60, color='steelblue', edgecolor='white')
axes[1,0].set_title('Price Distribution')
axes[1,0].set_xlabel('Price (BRL)')
axes[1,0].set_ylabel('Count')

# 4. Orders by Day of Week
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_counts = df['day_of_week'].value_counts().reindex(dow_order)
axes[1,1].bar(dow_counts.index, dow_counts.values, color='mediumseagreen')
axes[1,1].set_title('Orders by Day of Week')
axes[1,1].tick_params(axis='x', rotation=45)

# 5. Orders by Hour
hour_counts = df['hour'].value_counts().sort_index()
axes[2,0].bar(hour_counts.index, hour_counts.values, color='mediumpurple')
axes[2,0].set_title('Orders by Hour of Day')
axes[2,0].set_xlabel('Hour')
axes[2,0].set_ylabel('Order Count')

# 6. Delivery Days Distribution
delivery = df['delivery_days'].dropna()
delivery = delivery[(delivery >= 0) & (delivery <= 60)]
axes[2,1].hist(delivery, bins=40, color='darkorange', edgecolor='white')
axes[2,1].set_title('Delivery Days Distribution')
axes[2,1].set_xlabel('Days')
axes[2,1].set_ylabel('Count')

plt.tight_layout()
plt.savefig("outputs/module1_eda.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n Module 1 Complete! Plot saved: outputs/module1_eda.png")
