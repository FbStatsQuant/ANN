# Artificial Neural Networks (ANNs): Structure and Objectives

## 1. Supervised learning setup
We observe data
\[
\{(x_i, y_i)\}_{i=1}^n, \quad x_i \in \mathbb{R}^p,
\]
with either:
- **Regression**: \(y_i \in \mathbb{R}\)
- **Binary classification**: \(y_i \in \{0,1\}\)

The objective is to learn a function
\[
f_\theta : \mathbb{R}^p \to \mathbb{R}
\]
(or to \([0,1]\) for classification) that generalizes beyond the training sample.

---

## 2. ANN as a parametric nonlinear model
A fully connected ANN defines a parametric family of functions indexed by weights and biases \(\theta\).

Let
\[
h^{(0)} = x.
\]
For \(\ell = 1, \dots, L\), define hidden layers as
\[
h^{(\ell)} = \phi\big( W^{(\ell)} h^{(\ell-1)} + b^{(\ell)} \big),
\]
and the output layer as
\[
f_\theta(x) = g\big( W^{(L+1)} h^{(L)} + b^{(L+1)} \big).
\]

Here:
- \(\phi\) is a nonlinear activation function (e.g. ReLU, LeakyReLU)
- \(g\) is the output activation (identity or sigmoid)
- \(\theta = \{W^{(\ell)}, b^{(\ell)}\}_{\ell=1}^{L+1}\)

Without nonlinear activations, the model collapses to a linear map regardless of depth.

---

## 3. Output layer and task dependence
The output activation encodes the task:

### Regression
\[
g(z) = z
\]
so that \(f_\theta(x) \in \mathbb{R}\).

### Binary classification
\[
g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}
\]
which maps logits to probabilities \(\mathbb{P}(Y=1 \mid x)\).

The architecture is identical across tasks; only \(g\) and the loss change.

---

## 4. Loss functions and statistical meaning
Training minimizes an empirical risk
\[
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell\big(y_i, f_\theta(x_i)\big).
\]

### Mean Squared Error (regression)
\[
\ell(y, \hat y) = (y - \hat y)^2.
\]
This corresponds to maximum likelihood under a homoskedastic Gaussian noise model.

### Binary Cross-Entropy (classification)
\[
\ell(y, \hat p) = -\big[y \log \hat p + (1-y) \log(1-\hat p)\big].
\]
This corresponds to a Bernoulli likelihood with a logit link.

Thus, ANN training can be viewed as likelihood-based estimation under implicit noise assumptions.

---

## 5. Capacity and expressiveness
ANNs are **high-capacity** models:
- Width increases representational richness
- Depth enables hierarchical feature composition

Universal approximation results guarantee existence of approximating networks for broad function classes, but do not address:
- sample efficiency
- optimization difficulty
- generalization behavior

These aspects are controlled by optimization and regularization, not architecture alone.

---

## 6. Feature scaling and preprocessing
Because ANNs rely on gradient-based optimization:
- feature scaling is essential
- unscaled inputs distort gradient magnitudes

For tabular data, standardization is typically required for stable training.

---

## 7. Scope and limitations
ANNs are well-suited for:
- smooth nonlinear structure
- moderately high-dimensional tabular data
- extensible pipelines (ensembles, hybrid models)

Limitations include:
- limited interpretability
- sensitivity to hyperparameters
- competition from tree-based methods in purely tabular settings

---

## 8. Relation to implementations in this repository
The classification and regression notebooks in this repository instantiate the framework above using:
- fully connected layers
- LeakyReLU activations
- task-appropriate losses

Subsequent lectures address how these models are trained and regularized in practice.

---

**Next:** Backpropagation and Optimization
