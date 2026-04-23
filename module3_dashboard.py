import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs", exist_ok=True)

# Load Data
df = pd.read_csv("outputs/clean_ecommerce.csv", parse_dates=['order_purchase_timestamp'])
rfm = pd.read_csv("outputs/rfm_segments.csv")
print(f"Data loaded: {df.shape}")

# KPIs
total_revenue = df['price'].sum()
total_orders = df['order_id'].nunique()
total_customers = df['customer_unique_id'].nunique()
avg_order_val = total_revenue / total_orders

print(f"\nKPIs:")
print(f"   Total Revenue  : BRL {total_revenue:,.2f}")
print(f"   Total Orders   : {total_orders:,}")
print(f"   Unique Customers: {total_customers:,}")
print(f"   Avg Order Value : BRL {avg_order_val:,.2f}")

# Prepare Chart Data
# Monthly revenue
df['month_str'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
monthly_rev = df.groupby('month_str')['price'].sum().reset_index()
monthly_rev.columns = ['month','revenue']

# Top 10 categories
top_cat = df.groupby('product_category_name_english')['price'].sum().nlargest(10).reset_index()
top_cat.columns = ['category','revenue']

# Orders by day of week
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['day_of_week'] = df['order_purchase_timestamp'].dt.day_name()
dow_counts = df['day_of_week'].value_counts().reindex(dow_order).reset_index()
dow_counts.columns = ['day','orders']

# Payment types
pay_type = df.groupby('payment_type')['price'].sum().reset_index()
pay_type.columns = ['payment','revenue']

# RFM segments
seg_counts = rfm['Segment'].value_counts().reset_index()
seg_counts.columns = ['segment','count']

# Monthly order count
monthly_orders = df.groupby('month_str')['order_id'].nunique().reset_index()
monthly_orders.columns = ['month','orders']

# Build Dashboard
fig = make_subplots(
    rows=4, cols=3,
    subplot_titles=[
        "Total Revenue", "Total Orders", "Avg Order Value",
        "Monthly Revenue Trend", "Monthly Order Count", "Top 10 Categories",
        "Orders by Day of Week", "Payment Method Split", "Customer Segments",
        "Price vs Freight Scatter", "Revenue Cumulative", "Delivery Days"
    ],
    specs=[
        [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "domain"}, {"type": "domain"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
    ],
    vertical_spacing=0.08,
    horizontal_spacing=0.08
)

COLORS = px.colors.qualitative.Plotly

# Row 1: KPI Indicators
fig.add_trace(go.Indicator(
    mode="number",
    value=total_revenue,
    number={"prefix": "BRL ", "valueformat": ",.0f"},
    title={"text": "Total Revenue"}
), row=1, col=1)

fig.add_trace(go.Indicator(
    mode="number",
    value=total_orders,
    number={"valueformat": ","},
    title={"text": "Total Orders"}
), row=1, col=2)

fig.add_trace(go.Indicator(
    mode="number",
    value=avg_order_val,
    number={"prefix": "BRL ", "valueformat": ",.2f"},
    title={"text": "Avg Order Value"}
), row=1, col=3)

# Row 2: Monthly Revenue
fig.add_trace(go.Scatter(
    x=monthly_rev['month'], y=monthly_rev['revenue'],
    mode='lines+markers', name='Revenue',
    line=dict(color='royalblue', width=2),
    marker=dict(size=5)
), row=2, col=1)

# Row 2: Monthly Orders
fig.add_trace(go.Bar(
    x=monthly_orders['month'], y=monthly_orders['orders'],
    name='Orders', marker_color='mediumseagreen'
), row=2, col=2)

# Row 2: Top Categories
fig.add_trace(go.Bar(
    x=top_cat['revenue'][::-1], y=top_cat['category'][::-1],
    orientation='h', name='Category Revenue',
    marker_color='coral'
), row=2, col=3)

# Row 3: Day of week
fig.add_trace(go.Bar(
    x=dow_counts['day'], y=dow_counts['orders'],
    name='Orders by Day', marker_color='mediumpurple'
), row=3, col=1)

# Row 3: Payment pie
fig.add_trace(go.Pie(
    labels=pay_type['payment'], values=pay_type['revenue'],
    name='Payment', hole=0.3
), row=3, col=2)

# Row 3: Segment pie
fig.add_trace(go.Pie(
    labels=seg_counts['segment'], values=seg_counts['count'],
    name='Segments', hole=0.3,
    marker=dict(colors=['#2ecc71','#3498db','#f39c12','#e74c3c','#95a5a6'])
), row=3, col=3)

# Row 4: Price vs Freight scatter
sample = df.sample(min(2000, len(df)), random_state=42)
fig.add_trace(go.Scatter(
    x=sample['price'], y=sample['freight_value'],
    mode='markers', name='Price vs Freight',
    marker=dict(size=4, color='steelblue', opacity=0.5)
), row=4, col=1)

# Row 4: Cumulative Revenue
monthly_rev['cumulative'] = monthly_rev['revenue'].cumsum()
fig.add_trace(go.Scatter(
    x=monthly_rev['month'], y=monthly_rev['cumulative'],
    mode='lines', fill='tozeroy', name='Cumulative Revenue',
    line=dict(color='darkorange')
), row=4, col=2)

# Row 4: Delivery days histogram
delivery = df['delivery_days'].dropna()
delivery = delivery[(delivery >= 0) & (delivery <= 60)]
fig.add_trace(go.Histogram(
    x=delivery, nbinsx=40, name='Delivery Days',
    marker_color='indianred'
), row=4, col=3)

# Layout
fig.update_layout(
    title=dict(
        text="<b>E-Commerce Analytics Dashboard</b>",
        x=0.5, font=dict(size=22)
    ),
    height=1200,
    showlegend=False,
    paper_bgcolor='white',
    plot_bgcolor='#f8f9fa',
    font=dict(family="Arial", size=11)
)

fig.update_xaxes(showgrid=True, gridcolor='lightgray', tickangle=45)
fig.update_yaxes(showgrid=True, gridcolor='lightgray')

# Save HTML dashboard
output_path = "outputs/module3_dashboard.html"
fig.write_html(output_path)
print(f"\n Module 3 Complete!")
print(f"   Dashboard saved: {output_path}")
print(f"   Open this file in your browser to see the interactive dashboard!")

fig.show()
