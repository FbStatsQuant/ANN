# Weighted Loss for Imbalanced Classification

## Introduction

Standard training minimizes the average loss uniformly across all samples. When class distributions are skewed — as they frequently are in real-world problems such as fraud detection, medical diagnosis, and rare event prediction — this uniform treatment creates a systematic bias: the model learns to favor majority classes because they dominate the gradient signal.

This lecture develops the mathematical foundation of **weighted loss functions**, establishes why they correct the imbalance problem, and examines their practical implications through the lens of both binary and multiclass classification.

---

## 1. The Imbalance Problem

### Setup

Consider a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ with $K$ classes, where $n_k = |\{i : y_i = k\}|$ denotes the count of class $k$. We say the dataset is **imbalanced** when class frequencies differ substantially, i.e., $n_k \ll n_j$ for some $k, j$.

Denote the class proportions:
$$\pi_k = \frac{n_k}{n}, \quad \sum_{k=1}^K \pi_k = 1$$

In a fraud detection dataset we might have $\pi_{\text{fraud}} = 0.01$ and $\pi_{\text{legit}} = 0.99$.

### Why Standard Training Fails

The standard empirical risk minimization objective is:

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(y_i, f_\theta(x_i))$$

This is equivalent to a **class-frequency-weighted** average:

$$\mathcal{L}(\theta) = \sum_{k=1}^K \pi_k \cdot \underbrace{\frac{1}{n_k}\sum_{i: y_i = k} \ell(k, f_\theta(x_i))}_{\text{average loss on class } k}$$

The loss surface is therefore dominated by the majority class. During gradient descent, the parameter updates are driven primarily by the abundant class, and the rare class contributes negligible gradient signal proportional to $\pi_k$.

**Consequence**: The optimizer finds a solution that minimizes loss on the majority class, often at the expense of the minority class. A model predicting "not fraud" for all inputs achieves 99% accuracy while being completely useless.

---

## 2. Weighted Binary Cross-Entropy

### Standard BCE Loss

For binary classification with $y_i \in \{0, 1\}$ and predicted probability $p_i = \sigma(f_\theta(x_i))$, the Binary Cross-Entropy loss for sample $i$ is:

$$\ell(y_i, p_i) = -\left[y_i \log p_i + (1 - y_i)\log(1 - p_i)\right]$$

This decomposes into two terms: the positive class term $-y_i \log p_i$ (active only when $y_i = 1$) and the negative class term $-(1-y_i)\log(1-p_i)$ (active only when $y_i = 0$).

### Introducing pos_weight

The **weighted BCE** introduces a scalar $w^+ > 0$ that scales the positive class contribution:

$$\ell_w(y_i, p_i) = -\left[w^+ \cdot y_i \log p_i + (1 - y_i)\log(1 - p_i)\right]$$

The epoch-level loss becomes:

$$\mathcal{L}_w(\theta) = \frac{1}{n}\sum_{i=1}^n \ell_w(y_i, p_i)$$

### Choosing $w^+$

The standard choice is the **inverse frequency ratio**:

$$w^+ = \frac{n^-}{n^+} = \frac{\pi^-}{\pi^+}$$

where $n^-$ and $n^+$ are the counts of the negative and positive class respectively.

**Intuition**: This choice makes the total contribution of each class to the loss equal. The positive class has $n^+$ samples each weighted by $w^+ = n^-/n^+$, so their total contribution is:

$$n^+ \cdot w^+ = n^+ \cdot \frac{n^-}{n^+} = n^-$$

Which matches the total contribution of the negative class. The optimizer now sees a **balanced gradient signal** from both classes.

### Effect on Gradients

For $\text{BCEWithLogitsLoss}$, the gradient with respect to the pre-activation logit $z_i$ is:

$$\frac{\partial \ell_w}{\partial z_i} = \begin{cases} (p_i - 1) \cdot w^+ & \text{if } y_i = 1 \\ p_i & \text{if } y_i = 0 \end{cases}$$

With $w^+ = 20$ (fraud example), a missed fraud case produces a gradient 20× larger than a missed non-fraud case, forcing the optimizer to prioritize correcting fraud misclassifications.

### Implementation

**PyTorch:**
```python
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
pos_weight = torch.tensor([neg / pos])

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**TensorFlow:**
```python
pos_weight = neg / pos
class_weight = {0: 1.0, 1: float(pos_weight)}

model.fit(..., class_weight=class_weight)
```

Note that `BCEWithLogitsLoss` combines Sigmoid and BCE into a single numerically stable operation, avoiding floating point underflow that can occur when composing `Sigmoid` followed by `log` separately.

---

## 3. Weighted Multiclass Cross-Entropy

### Standard Categorical Cross-Entropy

For $K$-class classification with one-hot target $y_i \in \{0,1\}^K$ and softmax probabilities $p_i \in \Delta^{K-1}$:

$$\ell(y_i, p_i) = -\sum_{k=1}^K y_{ik} \log p_{ik} = -\log p_{i, y_i}$$

The last equality holds because only the true class term is nonzero.

### Weighted Extension

The **weighted categorical cross-entropy** assigns a per-class weight $w_k > 0$:

$$\ell_w(y_i, p_i) = -w_{y_i} \log p_{i, y_i}$$

The epoch-level objective becomes:

$$\mathcal{L}_w(\theta) = -\frac{1}{n}\sum_{i=1}^n w_{y_i} \log p_{i, y_i}$$

### Computing Class Weights for $K$ Classes

The standard **inverse frequency** formula generalizes directly:

$$w_k = \frac{n}{K \cdot n_k}$$

**Derivation**: We want the total weighted contribution of each class to be equal. Class $k$ has $n_k$ samples, so its total contribution without weighting is proportional to $n_k$. Setting:

$$w_k \propto \frac{1}{n_k}$$

and normalizing so that $\frac{1}{K}\sum_k w_k = 1$ yields:

$$w_k = \frac{n}{K \cdot n_k}$$

**Properties**:
- Majority class ($n_k$ large): $w_k < 1$, loss contribution reduced
- Minority class ($n_k$ small): $w_k \gg 1$, loss contribution amplified
- If all classes are balanced: $w_k = 1$ for all $k$, recovering standard CE

**Example** with 3 classes and 100,000 total samples:

| Class | Count | Weight |
|-------|-------|--------|
| 0 | 70,000 | $100000 / (3 \times 70000) = 0.476$ |
| 1 | 29,000 | $100000 / (3 \times 29000) = 1.149$ |
| 2 | 1,000  | $100000 / (3 \times 1000) = 33.33$  |

### Implementation

**PyTorch:**
```python
total = len(y_train)
K = 7  # number of classes

weights = []
for k in range(K):
    nk = (y_train == k).sum()
    weights.append(total / (K * nk))

class_weights = torch.tensor(weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**TensorFlow:**
```python
total = len(y_train)
K = 7

class_weight = {}
for k in range(K):
    nk = (y_train == k).sum()
    class_weight[k] = total / (K * nk)

model.fit(..., class_weight=class_weight)
```

---

## 4. The Precision-Recall Tradeoff

Weighted loss does not come free. Amplifying the minority class signal forces the model to be more sensitive to that class, which has a direct cost on specificity.

For binary classification, the confusion matrix entries shift as $w^+$ increases:

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **True Negative** | TN ↓ | FP ↑ |
| **True Positive** | FN ↓ | TP ↑ |

More minority class weight → more true positives and more false positives simultaneously.

The relevant metrics are:

$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$

These are in tension: as $w^+$ increases, recall improves and precision degrades. The **F1 score** combines both:

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

The optimal $w^+$ depends entirely on the cost structure of the application:
- **Fraud detection**: missing fraud (FN) is expensive → prioritize recall, accept lower precision
- **Medical screening**: false alarms (FP) cause unnecessary procedures → balance precision and recall
- **Spam filtering**: false positives (legitimate email flagged) are very costly → prioritize precision

---

## 5. Why It Is Not the Same Model

A common misconception is that weighted loss simply rescales the objective and produces an equivalent model. This is incorrect.

The architecture and hypothesis class are identical, but **the gradient trajectory through parameter space is different**. At each step $t$, the update:

$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta \mathcal{L}_w(\theta^{(t)})$$

follows a path determined by $w$. The weighted loss defines a **different loss landscape** — specifically, one where the "valleys" corresponding to minority class errors are deeper. Gradient descent follows this modified landscape and converges to a different stationary point $\theta^*_w \neq \theta^*$.

The resulting models have different decision boundaries in input space, different probability calibration, and different performance profiles across classes.

---

## 6. Alternative Approaches to Imbalance

Weighted loss is one tool among several. Alternatives and complements include:

**Resampling:**
- **Oversampling**: Duplicate minority class samples or generate synthetic ones (SMOTE)
- **Undersampling**: Remove majority class samples to balance counts
- Both modify the data distribution rather than the loss function

**Threshold adjustment**: Post-training, shift the decision threshold below 0.5 to increase sensitivity to the minority class without retraining

**Focal Loss**: A dynamic weighting scheme where the weight of each sample depends on the model's current confidence, down-weighting easy examples and focusing on hard ones:

$$\ell_{\text{focal}}(y_i, p_i) = -\alpha (1 - p_i)^\gamma \log p_i$$

where $\gamma > 0$ controls the focusing strength.

---

## 7. Summary

1. **Standard cross-entropy** implicitly weights classes by frequency, causing the optimizer to prioritize majority classes and ignore minority ones.

2. **Weighted cross-entropy** introduces explicit per-class scalars $w_k$ that amplify the gradient contribution of underrepresented classes, correcting the imbalance at the loss level.

3. The **inverse frequency** formula $w_k = n / (K \cdot n_k)$ is the standard choice, equating total class contributions to the loss.

4. Weighted loss produces a **different model** — not a rescaling of the same model — because it alters the gradient trajectory and the loss landscape.

5. The gain in minority class recall comes at a **cost in precision** on those classes, and the optimal tradeoff is determined by the application's cost structure.

6. For extreme imbalance (below ~1%), weighting alone may be insufficient and should be combined with resampling or architectural choices like Focal Loss.

---

## References

- King, G., & Zeng, L. (2001). Logistic regression in rare events data. *Political Analysis*.
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV*.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*.
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*.
