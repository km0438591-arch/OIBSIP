# Kanchan Maurya - OIBSIP Task 2 - Customer Segmentation - Without sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Hello! I am Kanchan Maurya - Task 2 Started")

# 1. Load
df = pd.read_csv("retail_sales.csv")
df.columns = df.columns.str.strip()
print(df.head())
print(df.info())
print("Missing:", df.isnull().sum().sum())
df = df.dropna()
print("Shape:", df.shape)

# Customer ID banao
df['Customer'] = df['Gender'] + "_" + df['Customer Age'].astype(str) + "_" + df['Product']
df['Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# 2. Descriptive
print(df.describe())
print("Avg Purchase:", df['Sales'].mean())
print("Total Sales:", df['Sales'].sum())
print(df.groupby('Product')['Sales'].sum())

# 3. RFM
max_date = df['Date'].max()
rfm = df.groupby('Customer').agg({
    'Date': lambda x: (max_date - x.max()).days,
    'Customer': 'count',
    'Sales': 'sum'
})
rfm.columns = ['Recency','Frequency','Monetary']
print(rfm.head(10))
print(rfm.describe())

# 4. StandardScaler - MANUAL - sklearn ke bina
for col in ['Recency','Frequency','Monetary']:
    rfm[col+'_scaled'] = (rfm[col] - rfm[col].mean()) / rfm[col].std()
print("Scaling Done!")

# 5. Elbow Method - MANUAL WCSS
def calc_wcss(data, k):
    # simple manual kmeans 1 iteration
    np.random.seed(42)
    centroids = data[np.random.choice(len(data), k, replace=True)]
    for _ in range(20):
        dist = np.sqrt(((data[:,None,:] - centroids[None,:,:])**2).sum(axis=2))
        labels = np.argmin(dist, axis=1)
        new_cent = np.array([data[labels==i].mean(axis=0) if np.any(labels==i) else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_cent):
            break
        centroids = new_cent
    wcss = np.sum((data - centroids[labels])**2)
    return wcss, labels, centroids

data_scaled = rfm[['Recency_scaled','Frequency_scaled','Monetary_scaled']].values
wcss = []
for k in range(1,4):
    w, _, _ = calc_wcss(data_scaled, k)
    wcss.append(w)
    print(f"K={k} WCSS={w:.2f}")

plt.figure(figsize=(8,5))
plt.plot(range(1,4), wcss, marker='o', color='purple', linewidth=2)
plt.title("Elbow Method - Kanchan Maurya")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig("Elbow_Method.png")
plt.show()
print("Graph 1 Elbow Done!")

# 6. KMeans MANUAL K=3
_, labels, _ = calc_wcss(data_scaled, 3)
rfm['Cluster'] = labels
print(rfm['Cluster'].value_counts())
print(rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean())

# 7. Scatter Frequency vs Monetary
plt.figure(figsize=(8,6))
sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Cluster', palette='Set2', s=150)
plt.title("Customer Segmentation - Frequency vs Monetary - Kanchan Maurya")
plt.savefig("Cluster_Frequency_Monetary.png")
plt.show()
print("Graph 2 Done!")

# 8. Scatter Recency vs Monetary
plt.figure(figsize=(8,6))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis', s=150)
plt.title("Customer Segmentation - Recency vs Monetary - Kanchan Maurya")
plt.savefig("Cluster_Recency_Monetary.png")
plt.show()
print("Graph 3 Done!")

# 9. Bar Chart
plt.figure(figsize=(7,5))
counts = rfm['Cluster'].value_counts().sort_index()
counts.plot(kind='bar', color=['#FF6B6B','#4ECDC4','#45B7D1'])
plt.title("Number of Customers per Cluster - Kanchan Maurya")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
for i, v in enumerate(counts):
    plt.text(i, v+0.1, str(v), ha='center', fontweight='bold')
plt.savefig("Customers_Per_Cluster.png")
plt.show()
print("Graph 4 Done!")

print("--- INSIGHTS by Kanchan Maurya ---")
for c in sorted(rfm['Cluster'].unique()):
    s = rfm[rfm['Cluster']==c]
    print(f"Cluster {c}: {len(s)} customers, Avg Monetary {s['Monetary'].mean():.2f}, Recency {s['Recency'].mean():.1f}")
print("High Monetary = VIP Customers - Premium Offer")
print("Low Recency = New Customers - Welcome Offer")
print("High Recency = Lost Customers - Re-engagement Offer")
print("Task 2 Completed by Kanchan Maurya!")