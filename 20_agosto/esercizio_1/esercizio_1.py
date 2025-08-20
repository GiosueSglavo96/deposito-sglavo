import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Leggo il dataset
df = pd.read_csv('C:\\Users\\KB316GR\\OneDrive - EY\\Desktop\\Dataset\\Mall Customers\\mall_Customers.csv')  # Sostituisci con il nome del tuo file

# Visualizzo le prime righe del dataset
print(df.head())

# Verifica duplicati su CustomerID
duplicati_id = df[df.duplicated(subset='CustomerID', keep=False)]
if not duplicati_id.empty:
    print("\nATTENZIONE: Sono presenti clienti con lo stesso CustomerID:")
    print(duplicati_id)
else:
    print("\nNon sono presenti clienti con lo stesso CustomerID.")

# Seleziono le colonne per il clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizzo i dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_clusters = 8

# Applico il clustering K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

#print(df.head(20))

# Visualizzo i cluster
plt.figure(figsize=(8, 6))
colors = cm.get_cmap('tab10', n_clusters) 
for i in range(n_clusters):
    cluster = df[df['Cluster'] == i]
    plt.scatter(cluster['Annual Income (k$)'], cluster['Spending Score (1-100)'],
                s=100, c=colors(i), label=f'Cluster {i}')

# Centroidi
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# Calcolo la distanza euclidea di ogni punto dal centroide del proprio cluster
centroids = kmeans.cluster_centers_
distances = np.linalg.norm(X_scaled - centroids[df['Cluster']], axis=1)

# Calcolo la distanza media
average_distance = distances.mean()
print(f"Distanza media dei punti dal centroide del proprio cluster: {average_distance:.4f}")