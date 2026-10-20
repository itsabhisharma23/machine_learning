# Principal Component Analysis (PCA) from Scratch

[PCA.py](PCA.py) implements PCA using only NumPy. Its `CustomPCA` class follows the same interface as scikit-learn's `PCA` (`fit`, `transform`, `fit_transform`, `components_`, `explained_variance_ratio_`) and gives the same results.

## What PCA does

Real datasets often have features that move together. If one feature is roughly 2.5 times another, storing both mostly repeats the same information. PCA finds new axes for the data, called **principal components**, such that:

1. The **first component** points in the direction where the data varies the most.
2. Each **next component** points in the direction of the most remaining variation, at right angles to all earlier ones.
3. The components are sorted by how much of the data's variance they capture.

Keeping only the first *k* components reduces the data from *n* features to *k* while losing as little information (variance) as possible. That's useful for compressing data, removing redundant features, visualizing high-dimensional data in 2D, and speeding up other models.

## How the code works

### Step 1: Center the data

```python
self.mean = np.mean(X, axis=0)
X_centered = X - self.mean
```

PCA measures variation around the mean, so each feature's average is subtracted first and every feature then has mean 0. Without this, the first component would point toward the data's average position instead of along its spread. The mean is saved so `transform` can center new data the same way.

### Step 2: Singular Value Decomposition (SVD)

```python
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
```

SVD splits the centered data matrix (*m* samples × *n* features) into three parts:

$$X_c = U \, S \, V^T$$

| Part | Shape | Meaning |
|---|---|---|
| `Vt` | *n* × *n* | Each **row** is a principal component: a direction in feature space, with length 1 |
| `S` | *n* values | Singular values, sorted from largest to smallest; how stretched the data is along each direction |
| `U` | *m* × *n* | The coordinates of each sample along each direction, scaled to length 1 |

`full_matrices=False` tells NumPy to skip the parts of `U` that are never used, which saves a lot of memory when there are many samples.

**Why SVD instead of eigenvectors?** The textbook recipe is to build the covariance matrix *C = X<sub>c</sub><sup>T</sup> X<sub>c</sub> / (m − 1)* and find its eigenvectors. Substituting the SVD shows why that gives the same answer:

$$C = \frac{X_c^T X_c}{m-1} = V \, \frac{S^2}{m-1} \, V^T$$

So the rows of `Vt` *are* the eigenvectors of the covariance matrix, and the eigenvalues are *S² / (m − 1)*. SVD gets there without forming *C*, which is faster and more numerically accurate. Squaring the data to form *C* loses precision. scikit-learn uses SVD for the same reason.

### Step 3: Explained variance ratio

```python
eigenvalues = (S ** 2) / (m - 1)
total_variance = np.sum(eigenvalues)
self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
```

Each eigenvalue is the **variance of the data along that component**. Dividing by the total gives the share of all variance each component captures. This is how you decide how many components to keep.

The eigenvalues add up to the sum of the individual feature variances. Rotating the axes doesn't create or destroy variance; it only redistributes it so the first few axes hold most of it.

### Step 4: Keep the top *k* components

```python
self.components_ = Vt[:self.n_components, :]
```

The SVD already sorts components by importance, so the first `n_components` rows are the *k* best directions. `components_` has shape (*k*, *n*).

### Projecting data: `transform`

```python
X_centered = X - self.mean
return X_centered @ self.components_.T
```

To get a sample's coordinates in the new space, center it with the **training** mean, then take the dot product with each component. The result has shape (*m*, *k*): each sample is now described by *k* numbers instead of *n*.

`fit_transform` simply calls `fit` and then `transform` on the same data.

## Usage

```python
import numpy as np
from PCA import CustomPCA

X = np.random.randn(100, 5)                 # 100 samples, 5 features

pca = CustomPCA(n_components=2)
X_reduced = pca.fit_transform(X)            # shape (100, 2)

print(pca.components_.shape)                # (2, 5): 2 directions in 5-D space
print(pca.explained_variance_ratio_)        # share of variance per component

X_new_reduced = pca.transform(np.random.randn(10, 5))   # reuse the training mean and components
```

## The demo

Run it with:

```bash
python3 PCA.py
```

The demo builds 500 samples with 5 features. The features form three independent groups:

| Feature | Built as | Group |
|---|---|---|
| `feature1` | random | A |
| `feature2` | 2.5 × `feature1` + small noise | A |
| `feature3` | random | B |
| `feature4` | −1.2 × `feature3` + some noise | B |
| `feature5` | random | C (independent) |

So the data really has 3 underlying sources of variation, not 5. Output:

```
Original shape: (500, 5)
Reduced shape: (500, 2)
Explained Variance Ratio by top 2 components: [0.65502027 0.24919894]
Total variance captured: 90.42%
```

Two components keep 90% of the variance. Looking inside them shows that PCA found the groups by itself:

| | feature1 | feature2 | feature3 | feature4 | feature5 | Variance |
|---|---|---|---|---|---|---|
| Component 1 | −0.37 | −0.93 | 0.04 | −0.06 | 0.00 | 65.5% |
| Component 2 | 0.02 | 0.06 | 0.60 | −0.80 | −0.02 | 24.9% |

- **Component 1 is group A.** It weights `feature1` and `feature2` with the same sign because they move together.
- **Component 2 is group B.** `feature3` and `feature4` have opposite signs because `feature4` is built with a negative multiplier.
- **Group C (`feature5`) is the third component**, at 8.7%. The fourth and fifth components together hold under 1%: they are just the small noise added to `feature2` and `feature4`.

## Things to know

- **Scale matters.** PCA follows variance, so features with larger values dominate. In the demo, `feature2` has a variance of about 6 while the others are around 1–2, which is why it gets the biggest weight (−0.93) in component 1. When features use different units (e.g. metres and dollars), standardize each one to mean 0 and standard deviation 1 before PCA.
- **A component's sign is arbitrary.** A direction and its exact opposite describe the same axis, so a component can come out as `v` or `-v`. Your first component has the opposite sign to scikit-learn's. Both are correct, but the projected values flip sign too. scikit-learn applies a sign convention so its results are repeatable; this implementation doesn't.
- **Choosing `n_components`.** A common rule is to keep the smallest *k* whose explained variance ratios add up to 90–95%. Since `fit` computes all the eigenvalues anyway, fitting with `n_components` equal to the number of features shows the whole picture.
- **Memory.** `np.linalg.svd` decomposes the full matrix. For very large datasets, a randomized SVD that only computes the top *k* components (e.g. `sklearn.utils.extmath.randomized_svd`) is much faster.

## Checked against scikit-learn

On the demo data, `CustomPCA(2)` matches `sklearn.decomposition.PCA(2)`:

| Output | Match |
|---|---|
| `explained_variance_ratio_` | identical |
| `components_` | identical up to sign |
| `transform(X)` | identical up to sign |
