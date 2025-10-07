import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("OnlineRetail.csv", parse_dates=["InvoiceDate"], encoding="ISO-8859-1")
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df

df = load_data()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters")
country_filter = st.sidebar.multiselect("Select Country", options=df['Country'].unique(), default=df['Country'].unique())
cluster_filter = st.sidebar.multiselect("Select Cluster", options=[0,1,2,3], default=[0,1,2,3])

filtered_df = df[df['Country'].isin(country_filter)]

# -------------------------
# RFM Analysis
# -------------------------
snapshot_date = filtered_df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = filtered_df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
}).rename(columns={"InvoiceDate":"Recency", "InvoiceNo":"Frequency", "TotalPrice":"Monetary"}).reset_index()

# -------------------------
# Fix dtypes for scaling
# -------------------------
X = rfm[["Recency","Frequency","Monetary"]].copy()
X["Recency"] = X["Recency"].astype(float)
X["Frequency"] = X["Frequency"].astype(float)
X["Monetary_log"] = np.log1p(X["Monetary"]).astype(float)

# Scale data & cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[["Recency","Frequency","Monetary_log"]])

km = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = km.fit_predict(X_scaled)
rfm = rfm[rfm["Cluster"].isin(cluster_filter)]

# -------------------------
# KPIs
# -------------------------
st.title("🛒 E-commerce Customer Analysis Dashboard")
st.header("Key Metrics")

total_customers = rfm['CustomerID'].nunique()
total_revenue = rfm['Monetary'].sum()
avg_clv = rfm['Monetary'].mean()
churned_customers = (rfm['Recency'] > 90).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", total_customers)
col2.metric("Total Revenue", f"${total_revenue:,.2f}")
col3.metric("Average CLV", f"${avg_clv:,.2f}")
col4.metric("Churned Customers", churned_customers)

# -------------------------
# Graphs
# -------------------------
st.header("📊 Visual Analysis")

# 1️⃣ Elbow & Silhouette Curve
inertias, sil_scores = [], []
for k in range(2,8):
    km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km_test.fit_predict(X_scaled)
    inertias.append(km_test.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].plot(range(2,8), inertias, "-o")
ax[0].set_title("Elbow Curve")
ax[1].plot(range(2,8), sil_scores, "-o")
ax[1].set_title("Silhouette Scores")
st.subheader("1️⃣ Choosing Number of Clusters")
st.pyplot(fig)

# 2️⃣ Number of Customers per Cluster
st.subheader("2️⃣ Number of Customers per Cluster")
fig, ax = plt.subplots()
sns.countplot(x="Cluster", data=rfm, palette="tab10", ax=ax)
st.pyplot(fig)

# 3️⃣ Average RFM per Cluster
st.subheader("3️⃣ Average RFM per Cluster")
fig, ax = plt.subplots()
rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean().plot(kind="bar", ax=ax)
st.pyplot(fig)

# 4️⃣ CLV Prediction
st.subheader("4️⃣ CLV Prediction: Actual vs Predicted")
y = rfm["Monetary"]
X_clv = rfm[["Recency", "Frequency", "Cluster"]]
X_clv = pd.get_dummies(X_clv, columns=["Cluster"], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_clv, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Monetary Value")
ax.set_ylabel("Predicted Monetary Value")
st.pyplot(fig)

# 5️⃣ Churn Prediction ROC Curve
st.subheader("5️⃣ Churn Prediction ROC Curve")
rfm["Churn"] = (rfm["Recency"] > 90).astype(int)
y = rfm["Churn"]
X_churn = rfm[["Recency","Frequency","Monetary","Cluster"]]
X_churn = pd.get_dummies(X_churn, columns=["Cluster"], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_churn, y, test_size=0.2, stratify=y, random_state=42)

clf = LogisticRegression(max_iter=200, class_weight="balanced")
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_proba)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
ax.plot([0,1],[0,1],'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve for Churn Prediction")
ax.legend()
st.pyplot(fig)
