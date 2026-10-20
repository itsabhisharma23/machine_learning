############################################
# Python code to implement PCA from scratch
############################################

import numpy as np

class CustomPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # Step 1: Mean center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Compute SVD of the centered data matrix 
        # (X_centered = U * S * Vt). 
        # Note: The rows of Vt are the eigenvectors of the covariance matrix.
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Step 3: Calculate explained variance ratio from singular values S
        # Eigenvalues of covariance matrix are proportional to S^2 / (m - 1)
        m = X.shape[0]
        eigenvalues = (S ** 2) / (m - 1)
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
        
        # Step 4: Store the top k principal components (rows of Vt correspond to principal axes)
        self.components_ = Vt[:self.n_components, :]
        
        return self

    def transform(self, X):
        # Project centered data onto the principal components
        X_centered = X - self.mean
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# --- Test the Implementation ---
if __name__ == "__main__":
    np.random.seed(42)
    # Generate synthetic dataset: 500 samples, 5 features with high correlation
    feature1 = np.random.randn(500)
    feature2 = feature1 * 2.5 + np.random.randn(500) * 0.2
    feature3 = np.random.randn(500)
    feature4 = feature3 * -1.2 + np.random.randn(500) * 0.5
    feature5 = np.random.randn(500)
    
    X_synthetic = np.column_stack((feature1, feature2, feature3, feature4, feature5))
    
    print(f"Original shape: {X_synthetic.shape}")
    
    # Initialize and fit our custom PCA down to 2 dimensions
    pca = CustomPCA(n_components=2)
    X_reduced = pca.fit_transform(X_synthetic)
    
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Explained Variance Ratio by top 2 components: {pca.explained_variance_ratio_}")
    print(f"Total variance captured: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")
