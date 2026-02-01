# Artificial Neural Networks (ANNs): Structure and Objectives

## Introduction

Artificial Neural Networks represent a fundamental class of parametric nonlinear models that have become central to modern machine learning. This lecture establishes the mathematical foundations of ANNs, their learning objectives, and their statistical interpretation.

---

## 1. The Supervised Learning Framework

We begin with the standard supervised learning setup. Consider observed data of the form:

$$\{(x_i, y_i)\}_{i=1}^n, \quad x_i \in \mathbb{R}^p$$

where the response variable $y_i$ takes one of two forms depending on our task:

- **Regression**: $y_i \in \mathbb{R}$
- **Binary classification**: $y_i \in \{0, 1\}$

The fundamental objective is to learn a function

$$f_\theta : \mathbb{R}^p \to \mathbb{R}$$

(or to $[0,1]$ for classification) that generalizes beyond the training sample. The key challenge in supervised learning is not merely fitting the training data, but discovering patterns that extend to unseen observations.

---

## 2. ANNs as Parametric Nonlinear Models

A fully connected ANN defines a parametric family of functions indexed by weights and biases $\theta$. The architecture proceeds through a sequence of transformations.

### Layer-wise Construction

Initialize with the input:
$$h^{(0)} = x$$

For each hidden layer $\ell = 1, \ldots, L$, we recursively define:

$$h^{(\ell)} = \phi\left(W^{(\ell)}h^{(\ell-1)} + b^{(\ell)}\right)$$

The output layer then produces:

$$f_\theta(x) = g\left(W^{(L+1)}h^{(L)} + b^{(L+1)}\right)$$

### Components

- $\phi$ is a **nonlinear activation function** (e.g., ReLU, LeakyReLU)
- $g$ is the **output activation** (identity for regression, sigmoid for classification)
- $\theta = \{W^{(\ell)}, b^{(\ell)}\}_{\ell=1}^{L+1}$ comprises all learnable parameters

### Critical Observation

**Without nonlinear activations, the model collapses to a linear map regardless of depth.** This is immediate from composition of affine transformations. The nonlinearity $\phi$ is what grants the network its expressive power.

---

## 3. Output Layer and Task Dependence

The choice of output activation $g$ encodes the statistical task and determines the interpretation of the network's output.

### Regression

For regression tasks:
$$g(z) = z$$

This identity activation ensures $f_\theta(x) \in \mathbb{R}$, allowing the network to predict continuous values without constraint.

### Binary Classification

For binary classification:
$$g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

The sigmoid function maps the real line to $(0,1)$, yielding:
$$f_\theta(x) = \mathbb{P}(Y = 1 \mid x)$$

This transformation converts logits into well-calibrated probabilities.

### Architectural Invariance

Notably, **the architecture is identical across tasks**—only $g$ and the loss function change. This modularity is a key design principle that separates the representation learning (hidden layers) from the task-specific prediction (output layer).

---

## 4. Loss Functions and Statistical Interpretation

Training minimizes an empirical risk:

$$\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(y_i, f_\theta(x_i))$$

The choice of loss $\ell$ implicitly encodes assumptions about the data-generating process.

### Mean Squared Error (Regression)

$$\ell(y, \hat{y}) = (y - \hat{y})^2$$

**Statistical interpretation**: This corresponds to maximum likelihood estimation under a homoskedastic Gaussian noise model:
$$Y \mid X \sim \mathcal{N}(f_\theta(X), \sigma^2)$$

### Binary Cross-Entropy (Classification)

$$\ell(y, \hat{p}) = -\left[y \log \hat{p} + (1-y)\log(1-\hat{p})\right]$$

**Statistical interpretation**: This corresponds to a Bernoulli likelihood with logit link:
$$Y \mid X \sim \text{Bernoulli}(\sigma(f_\theta(X)))$$

### Key Insight

ANN training can thus be viewed as **likelihood-based estimation under implicit noise assumptions**. The architecture defines a flexible parametric family, while the loss function specifies the probabilistic model for the data.

---

## 5. Capacity and Expressiveness

ANNs are **high-capacity models** characterized by:

- **Width increases representational richness**: More neurons per layer expand the hypothesis space
- **Depth enables hierarchical feature composition**: Successive layers can learn increasingly abstract representations

### Universal Approximation

Classical results guarantee that sufficiently wide networks can approximate any continuous function on compact domains. However, these theoretical guarantees **do not address**:

- **Sample efficiency**: How much data is required?
- **Optimization difficulty**: Can gradient descent find good solutions?
- **Generalization behavior**: Will the network perform well on unseen data?

### Practical Reality

These aspects are controlled by **optimization algorithms and regularization**, not architecture alone. Universal approximation tells us what's possible in principle, not what's achievable in practice with finite data and computational resources.

---

## 6. Feature Scaling and Preprocessing

Because ANNs rely on gradient-based optimization, preprocessing is critical.

### Why Scaling Matters

- **Feature scaling is essential**: Unscaled inputs distort gradient magnitudes
- **Unstable training**: Features on different scales can lead to ill-conditioned optimization landscapes

### Standard Practice

For tabular data, **standardization is typically required** for stable training:
$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

This ensures all features contribute comparably to gradient updates and prevents numerical issues during backpropagation.

---

## 7. Scope and Limitations

### Where ANNs Excel

ANNs are well-suited for:

- **Smooth nonlinear structure**: When relationships are continuous and differentiable
- **Moderately high-dimensional tabular data**: Where feature interactions matter
- **Extensible pipelines**: Ensembles, hybrid models, transfer learning

### Known Limitations

- **Limited interpretability**: Black-box nature makes inference difficult
- **Sensitivity to hyperparameters**: Architecture and training choices significantly impact performance
- **Competition from tree-based methods**: Gradient boosting often dominates on purely tabular tasks

The appropriate model choice depends critically on the problem structure, data characteristics, and interpretability requirements.

---

## 8. Implementation in This Repository

The classification and regression notebooks in this repository instantiate the framework presented here using:

- **Fully connected layers**: Standard dense architecture
- **LeakyReLU activations**: Hidden layer nonlinearity with $\phi(z) = \max(0.01z, z)$
- **Task-appropriate losses**: MSE for regression, BCE for classification

### What's Next

Subsequent materials will address:

- **Backpropagation**: Efficient computation of gradients via the chain rule
- **Optimization**: Stochastic gradient descent and adaptive methods
- **Regularization**: Techniques to control generalization (dropout, weight decay, early stopping)

---

## References and Further Reading

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*.
- Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*.

---

**Next Lecture**: Backpropagation and Optimization
