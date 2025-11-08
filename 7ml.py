import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names
df_original = pd.DataFrame(X, columns=feature_names)
df_original['target'] = y
print("Original Dataset (first 5 rows):")
print(df_original.head())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cov_matrix = np.cov(X_scaled, rowvar=False)
print("\nCovariance Matrix:")
print(pd.DataFrame(cov_matrix, columns=feature_names, index=feature_names))
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues (not sorted):")
print(eigenvalues)
print("\nEigenvectors (columns are eigenvectors, not sorted):")
print(pd.DataFrame(eigenvectors, columns=[f'EV{i+1}' for i in range(len(eigenvalues))], index=feature_names))
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
print("\nSorted Eigenvalues:")
print(sorted_eigenvalues)
print("\nSorted Eigenvectors:")
print(pd.DataFrame(sorted_eigenvectors, columns=[f'PC{i+1}' for i in range(len(eigenvalues))], index=feature_names))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
reduced_df = pd.DataFrame(X_pca, columns=['Principal Component 1', 'Principal Component 2']) 
reduced_df['target'] = y
print("\nReduced Dataset after PCA (first 5 rows):")
print(reduced_df.head())
print("\nPCA explained variance (from sklearn PCA):")
print(pca.explained_variance_)
print("\nPCA explained variance ratio (from sklearn PCA):")
print(pca.explained_variance_ratio_)
print("\nCumulative explained variance ratio:")
print(np.cumsum(pca.explained_variance_ratio_))
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
for label in np.unique(y):
    plt.scatter(
        X_pca[y == label, 0], 
        X_pca[y == label, 1], 
        label=target_names[label]
    )
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA on Iris Dataset")
plt.legend()
plt.show()
