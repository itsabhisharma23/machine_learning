# Linear Regression from Scratch

A from-scratch implementation of linear regression trained with batch gradient
descent, using only `numpy`/`pandas`. Results are cross-checked against
scikit-learn's `LinearRegression` on the same data.

## Contents

- `LinearRegressor.py`
  - `StandardScaler` — standardizes features to zero mean / unit variance
    using z-scores (`(X - mean) / std`), with a guard against divide-by-zero
    for constant columns.
  - `LinearRegressor` — fits `y = Xw + b` via batch gradient descent on mean
    squared error.

## How it works

**Cost function** (mean squared error, halved for a cleaner gradient):

```
cost = (1 / 2m) * sum((y_hat - y)^2)
```

**Gradients:**

```
dw = (1/m) * X.T @ (y_hat - y)
db = (1/m) * sum(y_hat - y)
```

**Update rule**, applied for a fixed number of iterations:

```
W = W - alpha * dw
b = b - alpha * db
```

Cost is logged every 100 iterations and stored in `cost_vector` as
`(iteration, cost)` tuples for later inspection/plotting.

## Usage

```python
from LinearRegressor import StandardScaler, LinearRegressor

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegressor()
model.fit(X_scaled, y, alpha=0.001, iterations=10000)

predictions = model.predict(X_scaled)
print(model.W, model.b)
```

## Running the demo

```bash
python LinearRegressor.py
```

This generates random dummy data, trains both the custom `LinearRegressor`
and scikit-learn's `LinearRegression` on the same scaled inputs, and prints
the learned coefficients/intercept from each side by side for comparison.

## Notes / limitations

- Uses plain batch gradient descent (no momentum, learning-rate schedule, or
  convergence check) — `alpha` and `iterations` must be tuned by hand.
- No regularization (L1/L2).
- No train/test split in the demo script; it's only meant to sanity-check the
  implementation against sklearn, not to evaluate generalization.
