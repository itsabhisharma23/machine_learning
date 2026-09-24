# Ridge and Lasso Regularization from Scratch

From-scratch implementations of Ridge (L2) and Lasso (L1) regression built with
`numpy`. Both are cross-checked against scikit-learn's `Ridge` and `Lasso` on
the same data.

## Contents

- `RidgeRegularizer.py`
  - `RidgeRegularizer`: fits Ridge regression with either the closed-form
    normal equation or batch gradient descent.
- `LassoRegularizer.py`
  - `LassoRegularizer`: fits Lasso regression with proximal gradient descent
    (gradient step followed by soft-thresholding).

## Why regularize?

Plain linear regression picks the weights that minimize training error and
nothing else. With many features, correlated features, or little data, the
weights can grow large and fit noise in the training set (overfitting).

Regularization adds a penalty on the size of the weights to the cost:

```
cost = data error (MSE) + lambda * penalty(W)
```

- `lambda` sets how strong the penalty is. `lambda = 0` is plain linear
  regression. Larger `lambda` pushes weights toward zero, trading a little
  bias for lower variance.
- The bias `b` is **not** penalized. It only shifts predictions up or down,
  and shrinking it would just pull predictions toward 0.
- Features should be on the same scale (standardized), because the penalty
  treats every weight equally. Both demos run `StandardScaler` first.

### Ridge (L2)

The penalty is the sum of squared weights:

```
penalty(W) = sum(W^2)
```

- Every weight shrinks in proportion to its size, but it rarely reaches
  exactly zero.
- Useful when most features carry some signal, or when features are
  correlated. Ridge spreads the weight across correlated features instead of
  picking one arbitrarily.
- The cost stays smooth and differentiable, so there is a closed-form
  solution.

### Lasso (L1)

The penalty is the sum of absolute weights:

```
penalty(W) = sum(|W|)
```

- Pushes weights toward zero by a **constant** amount, so small weights land
  at exactly `0`. Lasso therefore also selects features: it produces sparse
  models.
- Useful when you expect only a few features to matter.
- `|w|` is not differentiable at `0`, so there is no closed-form solution and
  plain gradient descent does not work well (weights oscillate around zero
  instead of settling there). This code uses proximal gradient descent
  instead.

### Side by side

|                      | Ridge (L2)                   | Lasso (L1)                          |
|----------------------|------------------------------|-------------------------------------|
| Penalty              | `sum(W^2)`                   | `sum(abs(W))`                       |
| Effect on weights    | Shrinks all, rarely to zero  | Shrinks, sets many to exactly zero  |
| Feature selection    | No                           | Yes                                 |
| Closed-form solution | Yes                          | No                                  |
| Solver in this repo  | Normal equation or GD        | Proximal GD (soft-thresholding)     |

## How the code works

In both classes, `alpha` is the **learning rate** and `lam` is the
**regularization strength** (`lambda`). scikit-learn names the regularization
strength `alpha`, so `lam` here corresponds to sklearn's `alpha`.

### `RidgeRegularizer.fit(X, y, alpha, lam, iterations=0)`

The `iterations` argument picks the solver.

**`iterations == 0`: normal equation (closed form).**

Setting the gradient of the Ridge cost to zero gives:

```
W = (X_b.T @ X_b + lambda * I')^-1 @ X_b.T @ y
```

- `X_b` is `X` with a column of ones prepended, so the first entry of the
  solution is the bias.
- `I'` is the identity matrix with `I'[0, 0] = 0`, so the bias is not
  penalized (`penality[0,0] = 0` in the code).
- Adding `lambda * I'` also makes `X_b.T @ X_b` invertible even when features
  are collinear, a useful side effect of Ridge.
- After solving, the first element is stored as `_b` and the rest as `_W`.

**`iterations > 0`: batch gradient descent.**

Cost:

```
cost = (1/2m) * sum((y_hat - y)^2) + (lambda/2m) * sum(W^2)
```

Gradients (the L2 term adds `lambda * W`, but nothing to the bias):

```
dW = (X.T @ (y_hat - y) + lambda * W) / m
db = sum(y_hat - y) / m
```

Update:

```
W = W - alpha * dW
b = b - alpha * db
```

The update can be rewritten as `W = W * (1 - alpha*lambda/m) - alpha * dMSE`.
Each step first multiplies the weights by a factor slightly below 1. This is
why L2 regularization is also called **weight decay**.

Cost is printed every 100 iterations.

### `LassoRegularizer.fit(X, y, alpha, lam, iterations=1000)`

Uses **proximal gradient descent** (also known as ISTA). Each iteration has
two steps:

1. **Gradient step on the MSE part only** (the smooth part):

   ```
   dW = X.T @ (y_hat - y) / m
   db = sum(y_hat - y) / m
   W_temp = W - alpha * dW
   ```

2. **Proximal step for the L1 part** (soft-thresholding, in
   `_soft_threshold`):

   ```
   W = sign(W_temp) * max(|W_temp| - alpha * lambda, 0)
   ```

   - Any weight with `|W_temp| <= alpha * lambda` is set to exactly `0`.
   - Every other weight moves toward zero by `alpha * lambda`.

   This step is how Lasso produces exact zeros. A plain subgradient update
   would only make weights bounce around zero.

The bias gets an ordinary gradient update and is never thresholded.

This update minimizes the same objective as `sklearn.linear_model.Lasso`:

```
(1/2m) * sum((y_hat - y)^2) + lambda * sum(|W|)
```

The cost is printed 5 times over the run.

### Shared methods

- `predict(X)`: returns `X @ W + b`. The Ridge version also takes an unused
  `y` argument: `predict(X, y)`.
- `get_weights()`: prints the learned weights and bias.

## Usage

```python
from sklearn.preprocessing import StandardScaler
from RidgeRegularizer import RidgeRegularizer
from LassoRegularizer import LassoRegularizer

X = StandardScaler().fit_transform(X)

# Ridge: closed form
ridge = RidgeRegularizer()
ridge.fit(X, y, alpha=0.001, lam=1)

# Ridge: gradient descent
ridge = RidgeRegularizer()
ridge.fit(X, y, alpha=0.001, lam=1, iterations=10000)
ridge.get_weights()

# Lasso: proximal gradient descent
lasso = LassoRegularizer()
lasso.fit(X, y, alpha=0.01, lam=0.1, iterations=5000)
lasso.get_weights()
```

## Running the demos

```bash
python3 RidgeRegularizer.py
python3 LassoRegularizer.py
```

Both demos generate random data (3000 rows, 39 features, seed 45),
standardize it, and compare against scikit-learn:

- **Ridge:** the normal equation's weights match `sklearn.linear_model.Ridge(alpha=1.0)`
  exactly. Gradient descent with 10,000 iterations agrees to about 3–4
  decimal places.
- **Lasso:** matches `sklearn.linear_model.Lasso(alpha=0.1)`. Every weight
  is exactly `0` and the bias is about `0.5009`. That result is expected: the
  target is random noise, unrelated to the features, so Lasso correctly
  drops all of them and predicts the mean of `y`.
