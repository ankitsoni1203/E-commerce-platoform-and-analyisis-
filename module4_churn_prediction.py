import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("outputs/clean_ecommerce.csv", parse_dates=['order_purchase_timestamp'])
print(f"✅ Loaded: {df.shape}")

SNAPSHOT_DATE = pd.Timestamp('2018-10-01')
CHURN_DAYS    = 180

last_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
last_purchase.columns = ['customer_unique_id', 'last_purchase']
last_purchase['days_since'] = (SNAPSHOT_DATE - last_purchase['last_purchase']).dt.days
last_purchase['churn'] = (last_purchase['days_since'] > CHURN_DAYS).astype(int)

print(f"\n📊 Churn Distribution:")
print(last_purchase['churn'].value_counts())
print(f"   Churn Rate: {last_purchase['churn'].mean()*100:.1f}%")

features = df.groupby('customer_unique_id').agg(
    total_orders      = ('order_id',               'nunique'),
    total_revenue     = ('price',                  'sum'),
    avg_order_value   = ('price',                  'mean'),
    max_order_value   = ('price',                  'max'),
    min_order_value   = ('price',                  'min'),
    total_items       = ('order_item_id',           'count'),
    avg_items_order   = ('order_item_id',           'mean'),
    total_freight     = ('freight_value',           'sum'),
    avg_freight       = ('freight_value',           'mean'),
    unique_categories = ('product_category_name_english', 'nunique'),
    avg_delivery_days = ('delivery_days',           'mean'),
    payment_installments = ('payment_installments', 'mean'),
    recency           = ('order_purchase_timestamp', lambda x: (SNAPSHOT_DATE - x.max()).days),
    days_active       = ('order_purchase_timestamp', lambda x: (x.max() - x.min()).days),
).reset_index()

features['purchase_frequency'] = features['total_orders'] / (features['days_active'] + 1)
features['revenue_per_order'] = features['total_revenue'] / features['total_orders']
features = features.merge(last_purchase[['customer_unique_id','churn']], on='customer_unique_id')

print(f"\n✅ Feature matrix: {features.shape}")
print(f"   Features: {list(features.columns[1:-1])}")

X = features.drop(columns=['customer_unique_id','churn'])
y = features['churn']

X.fillna(X.median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Train: {X_train.shape} | Test: {X_test.shape}")

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5,
        learning_rate=0.1, random_state=42
    )
}

results = {}
print("\n🚀 Training models...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_prob)
    cv_auc  = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    results[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob, 'auc': auc, 'cv_auc': cv_auc}
    print(f"   {name}")
    print(f"     Test AUC : {auc:.4f}")
    print(f"     CV AUC   : {cv_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Active','Churned']))

best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name} | AUC = {best['auc']:.4f}")

rf_model   = results['Random Forest']['model']
feat_imp   = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = feat_imp.head(12)

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle('Churn Prediction — Model Results', fontsize=15, fontweight='bold')

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0,0].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
axes[0,0].plot([0,1],[0,1],'k--', linewidth=1)
axes[0,0].set_title('ROC Curves')
axes[0,0].set_xlabel('False Positive Rate')
axes[0,0].set_ylabel('True Positive Rate')
axes[0,0].legend(fontsize=8)

cm = confusion_matrix(y_test, best['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Active','Churned'])
disp.plot(ax=axes[0,1], colorbar=False, cmap='Blues')
axes[0,1].set_title(f'Confusion Matrix — {best_name}')

names  = list(results.keys())
aucs   = [results[n]['auc'] for n in names]
cv_aucs= [results[n]['cv_auc'] for n in names]
x      = np.arange(len(names))
w      = 0.35
axes[0,2].bar(x - w/2, aucs,    w, label='Test AUC',  color='steelblue')
axes[0,2].bar(x + w/2, cv_aucs, w, label='CV AUC',    color='coral')
axes[0,2].set_xticks(x)
axes[0,2].set_xticklabels(names, rotation=15, ha='right')
axes[0,2].set_ylim(0.5, 1.0)
axes[0,2].set_title('Model Comparison — AUC')
axes[0,2].legend()

axes[1,0].barh(top_features.index[::-1], top_features.values[::-1], color='mediumseagreen')
axes[1,0].set_title('Feature Importance (Random Forest)')
axes[1,0].set_xlabel('Importance')

axes[1,1].hist(best['y_prob'][y_test==0], bins=40, alpha=0.6, label='Active',  color='steelblue')
axes[1,1].hist(best['y_prob'][y_test==1], bins=40, alpha=0.6, label='Churned', color='coral')
axes[1,1].set_title('Predicted Probability Distribution')
axes[1,1].set_xlabel('Churn Probability')
axes[1,1].legend()

axes[1,2].scatter(
    features[features['churn']==0]['recency'],
    features[features['churn']==0]['total_orders'],
    alpha=0.3, s=10, label='Active', color='steelblue'
)
axes[1,2].scatter(
    features[features['churn']==1]['recency'],
    features[features['churn']==1]['total_orders'],
    alpha=0.3, s=10, label='Churned', color='coral'
)
axes[1,2].set_title('Recency vs Orders (Active vs Churned)')
axes[1,2].set_xlabel('Recency (days)')
axes[1,2].set_ylabel('Total Orders')
axes[1,2].legend()

plt.tight_layout()
plt.savefig("outputs/module4_churn.png", dpi=150, bbox_inches='tight')
plt.show()

features['churn_probability'] = results['Random Forest']['model'].predict_proba(X)[:, 1]
features[['customer_unique_id','recency','total_orders','total_revenue','churn','churn_probability']]\
    .to_csv("outputs/churn_predictions.csv", index=False)

print("\n✅ Module 4 Complete!")
print(f"   Plot saved  : outputs/module4_churn.png")
print(f"   Predictions : outputs/churn_predictions.csv")
