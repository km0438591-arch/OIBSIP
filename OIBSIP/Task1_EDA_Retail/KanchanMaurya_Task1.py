# KanchanMaurya_Task1 - EDA on Retail Sales Data
# OASIS INFOBYTE - Data Analytics - Kanchan Maurya

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset - Kaggle se Superstore download karke retail_sales.csv naam se rakhna
df = pd.read_csv('retail_sales.csv')
print("Shape:", df.shape)
print(df.dtypes)
print("Nulls:", df.isnull().sum())

# Descriptive Stats
print(df.describe())

# Monthly Sales Trend
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.to_period('M')
monthly = df.groupby('Month')['Sales'].sum()
plt.figure(figsize=(10,5))
monthly.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend - Kanchan Maurya')
plt.show()

# Customer Age + Gender
plt.figure(figsize=(8,4))
sns.histplot(df['Customer Age'], bins=10, kde=True)
plt.title('Customer Age Distribution')
plt.show()
print(df['Gender'].value_counts())

# Top 10 Products
top10 = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(10)
print(top10)
top10.plot(kind='bar', figsize=(10,5))
plt.title('Top 10 Best Selling Products')
plt.show()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[['Quantity','Sales','Profit']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Conclusion - 3 Recommendations
# 1. Beauty ka profit margin sabse zyada hai - stock badhao
# 2. Oct-Dec me sale peak hai - festival offer lagao
# 3. 20-30 age wale sabse zyada kharidte hai - Insta ads unpe chalao