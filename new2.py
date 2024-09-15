import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('customer_data.csv')

df.dropna(inplace=True)  
df['total_spent'] = df['purchase_history'].apply(lambda x: sum(map(float, x.split(',')))) 
toprods = df['most_browsed_product'].value_counts().head(10)
tocateg = df['most_browsed_category'].value_counts().head(5)

# Visualize top products
plt.figure(figsize=(10, 6))
sns.barplot(x=toprods.index, y=toprods.values)
plt.title('Top 10 product')
plt.xlabel('prodt')
plt.ylabel('Number of Browses')
plt.xticks(rotation=45)
plt.show()

X = df[['total_spent', 'purchase_frequency', 'browsing_duration']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['customer_segment'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_spent', y='purchase_frequency', hue='customer_segment', data=df, palette='Set1')
plt.title('Customer Frequency')
plt.xlabel('Total Spent')
plt.ylabel('Purchase Frequency')
plt.show()


