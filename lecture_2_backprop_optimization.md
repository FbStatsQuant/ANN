# Backpropagation and Optimization

## Introduction

Having established the architectural framework of neural networks, we now address the computational machinery that makes training feasible. The key insight is that **backpropagation** enables efficient gradient computation through the chain rule, while **optimization algorithms** navigate the high-dimensional parameter space toward effective solutions.

This lecture develops the mathematical foundations of both components and examines their practical implications.

---

## 1. The Optimization Problem

Recall that training minimizes the empirical risk:

$$\min_{\theta} \mathcal{L}(\theta) = \min_{\theta} \frac{1}{n}\sum_{i=1}^n \ell(y_i, f_\theta(x_i))$$

where $\theta = \{W^{(\ell)}, b^{(\ell)}\}_{\ell=1}^{L+1}$ comprises all network parameters.

### Dimensionality

For a network with architecture $[d_0, d_1, \ldots, d_{L+1}]$, the total parameter count is:

$$|\theta| = \sum_{\ell=1}^{L+1} d_\ell \cdot d_{\ell-1} + \sum_{\ell=1}^{L+1} d_\ell$$

Modern networks routinely contain millions or billions of parameters, making closed-form solutions infeasible and necessitating iterative gradient-based methods.

### Why Gradients?

The loss landscape $\mathcal{L}(\theta)$ is highly nonconvex due to the composition of nonlinear activations. Despite lacking global optimality guarantees, **gradient descent remains the workhorse** because:

- It scales to high dimensions
- It exploits local geometric information
- It can escape saddle points via stochasticity
- Empirically, local minima found by gradient methods generalize well

---

## 2. The Backpropagation Algorithm

Backpropagation is not a distinct optimization algorithm—it's an **efficient procedure for computing gradients** via the chain rule. The name derives from the backward pass through the computational graph.

### Forward Pass

For a given input $x_i$, we compute:

$$
\begin{align}
h^{(0)} &= x_i \\
z^{(\ell)} &= W^{(\ell)}h^{(\ell-1)} + b^{(\ell)} \quad &\text{(pre-activation)} \\
h^{(\ell)} &= \phi(z^{(\ell)}) \quad &\text{(post-activation)} \\
\hat{y}_i &= g(z^{(L+1)})
\end{align}
$$

This forward pass produces the prediction $\hat{y}_i$ and caches all intermediate values $\{z^{(\ell)}, h^{(\ell)}\}$.

### Backward Pass: Computing Gradients

We need $\nabla_\theta \mathcal{L}$, which decomposes as gradients with respect to each weight matrix and bias vector.

**Step 1: Output gradient**

Define the error at the output:
$$\delta^{(L+1)} = \frac{\partial \ell}{\partial z^{(L+1)}} = \frac{\partial \ell}{\partial \hat{y}} \cdot g'(z^{(L+1)})$$

For MSE loss with identity output: $\delta^{(L+1)} = \hat{y}_i - y_i$

For BCE with sigmoid output: $\delta^{(L+1)} = \hat{y}_i - y_i$ (remarkably, the same form!)

**Step 2: Backpropagate errors**

For $\ell = L, L-1, \ldots, 1$:

$$\delta^{(\ell)} = \left[(W^{(\ell+1)})^T \delta^{(\ell+1)}\right] \odot \phi'(z^{(\ell)})$$

where $\odot$ denotes element-wise multiplication. This recursion propagates gradients backward through the network.

**Step 3: Compute parameter gradients**

Once we have all $\delta^{(\ell)}$, the gradients are:

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} &= \delta^{(\ell)} (h^{(\ell-1)})^T \\
\frac{\partial \mathcal{L}}{\partial b^{(\ell)}} &= \delta^{(\ell)}
\end{align}
$$

### Computational Complexity

- **Forward pass**: $O(|\theta|)$ operations
- **Backward pass**: $O(|\theta|)$ operations

Crucially, **computing all gradients costs only 2-3× a forward pass**, not $O(|\theta|^2)$ as naive finite differences would require. This efficiency makes training deep networks tractable.

---

## 3. Activation Function Derivatives

The backward pass requires derivatives of activation functions. Common choices:

### ReLU
$$\phi(z) = \max(0, z), \quad \phi'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

**Properties**: Dead neurons (zero gradient when inactive), no gradient saturation for positive inputs.

### LeakyReLU
$$\phi(z) = \max(\alpha z, z), \quad \phi'(z) = \begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases}$$

where typically $\alpha = 0.01$. **Advantage**: Prevents completely dead neurons.

### Sigmoid
$$\sigma(z) = \frac{1}{1+e^{-z}}, \quad \sigma'(z) = \sigma(z)(1-\sigma(z))$$

**Issue**: Gradient vanishing for $|z| \gg 0$, problematic for deep networks.

### Tanh
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)$$

**Properties**: Zero-centered outputs, but still suffers from saturation.

### Modern Default

**ReLU and its variants (LeakyReLU, PReLU, ELU)** have become standard for hidden layers due to:
- Computational efficiency
- Reduced vanishing gradient issues
- Empirical performance

---

## 4. Gradient Descent and Variants

Armed with gradients, we now discuss how to use them for optimization.

### Batch Gradient Descent

The classical update:
$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta \mathcal{L}(\theta^{(t)})$$

where $\eta > 0$ is the **learning rate**.

**Pros**: Stable, deterministic updates
**Cons**: Requires full dataset pass per update—prohibitively expensive for large $n$

### Stochastic Gradient Descent (SGD)

Instead of computing the exact gradient over all $n$ samples, we approximate using a **mini-batch** $\mathcal{B}_t \subset \{1, \ldots, n\}$:

$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta \frac{1}{|\mathcal{B}_t|}\sum_{i \in \mathcal{B}_t} \ell(y_i, f_\theta(x_i))$$

**Mini-batch size**: Typically 32, 64, 128, or 256.

**Key properties**:
- **Computational efficiency**: Updates are fast
- **Stochasticity**: Noise helps escape saddle points and sharp minima
- **Scalability**: Memory footprint independent of dataset size

The stochastic gradient is an unbiased estimator of the true gradient:
$$\mathbb{E}_{\mathcal{B}}[\nabla_{\mathcal{B}} \mathcal{L}] = \nabla \mathcal{L}$$

### Momentum

Plain SGD can oscillate in ravines where gradients are steep in some directions but shallow in others. **Momentum** accumulates a velocity vector:

$$
\begin{align}
v^{(t+1)} &= \beta v^{(t)} + \nabla_\theta \mathcal{L}_{\mathcal{B}_t}(\theta^{(t)}) \\
\theta^{(t+1)} &= \theta^{(t)} - \eta v^{(t+1)}
\end{align}
$$

where $\beta \in [0, 1)$ (typically 0.9) controls the decay.

**Effect**: Smooths updates, accelerates convergence in consistent directions, dampens oscillations.

### Nesterov Momentum

A variant that "looks ahead":

$$
\begin{aligned}
v^{(t+1)} &= \beta v^{(t)} + \nabla_{\theta} \mathcal{L}_{\mathcal{B}_t}(\theta^{(t)} - \eta \beta v^{(t)}) \\
\theta^{(t+1)} &= \theta^{(t)} - \eta v^{(t+1)}
\end{aligned}
$$

Evaluates the gradient at a predicted future position, often providing faster convergence.

---

## 5. Adaptive Learning Rate Methods

Fixed learning rates are problematic: too large causes divergence, too small slows convergence. **Adaptive methods** adjust learning rates per parameter based on gradient history.

### AdaGrad

$$
\begin{align}
G^{(t+1)} &= G^{(t)} + (g^{(t)})^2 \\
\theta^{(t+1)} &= \theta^{(t)} - \frac{\eta}{\sqrt{G^{(t+1)} + \epsilon}} \odot g^{(t)}
\end{align}
$$

where $g^{(t)} = \nabla_\theta \mathcal{L}_{\mathcal{B}_t}(\theta^{(t)})$ and $G^{(t)}$ accumulates squared gradients.

**Issue**: Learning rate monotonically decreases, often becoming too small.

### RMSProp

Fixes AdaGrad's aggressive decay via exponential moving average:

$$
\begin{aligned}
G^{(t+1)} &= \gamma G^{(t)} + (1-\gamma)(g^{(t)})^2 \\
\theta^{(t+1)} &= \theta^{(t)} - \frac{\eta}{\sqrt{G^{(t+1)} + \epsilon}} \circ g^{(t)}
\end{aligned}
$$

with $\gamma \approx 0.9$. Discounts old gradients, maintaining adaptivity throughout training.

### Adam (Adaptive Moment Estimation)

Combines momentum and RMSProp:

$$
\begin{align}
m^{(t+1)} &= \beta_1 m^{(t)} + (1-\beta_1) g^{(t)} \quad &\text{(first moment)} \\
v^{(t+1)} &= \beta_2 v^{(t)} + (1-\beta_2) (g^{(t)})^2 \quad &\text{(second moment)} \\
\hat{m}^{(t+1)} &= \frac{m^{(t+1)}}{1-\beta_1^{t+1}} \quad &\text{(bias correction)} \\
\hat{v}^{(t+1)} &= \frac{v^{(t+1)}}{1-\beta_2^{t+1}} \quad &\text{(bias correction)} \\
\theta^{(t+1)} &= \theta^{(t)} - \eta \frac{\hat{m}^{(t+1)}}{\sqrt{\hat{v}^{(t+1)}} + \epsilon}
\end{align}
$$

**Default hyperparameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.

**Status**: Adam has become the de facto standard optimizer for many applications due to robustness and minimal tuning requirements.

---

## 6. Learning Rate Schedules

Even with adaptive methods, global learning rate scheduling can improve convergence.

### Step Decay
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

Reduce learning rate by factor $\gamma$ every $s$ epochs.

### Exponential Decay
$$\eta_t = \eta_0 e^{-\lambda t}$$

Smooth exponential reduction.

### Cosine Annealing
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$

Smoothly decreases from $\eta_{\max}$ to $\eta_{\min}$ over $T$ iterations.

### Warm Restarts

Periodically reset learning rate to high value, allowing escape from local minima.

### Learning Rate Warmup

Start with very small learning rate and gradually increase to target value over first few epochs. Stabilizes training, especially for large batch sizes.

---

## 7. Challenges in Optimization

### Vanishing Gradients

In deep networks, repeated multiplication of gradients can cause:
$$\|\delta^{(1)}\| \ll \|\delta^{(L)}\|$$

**Consequence**: Early layers receive negligible updates, learning stalls.

**Solutions**:
- ReLU activations (mitigate but don't eliminate)
- Batch normalization
- Residual connections (ResNets)
- Careful initialization (Xavier, He)

### Exploding Gradients

Conversely, gradients can grow exponentially:
$$\|\delta^{(1)}\| \gg \|\delta^{(L)}\|$$

**Solutions**:
- Gradient clipping: $g \leftarrow g \cdot \min(1, \theta/\|g\|)$
- Weight regularization
- Batch normalization

### Saddle Points

High-dimensional loss landscapes contain exponentially many saddle points but relatively few local minima. 

**Observation**: Stochasticity in SGD helps escape saddle points, as noise can push parameters off zero-gradient plateaus.

### Ill-Conditioning

If the Hessian has widely varying eigenvalues, optimization becomes difficult. Adaptive methods partially address this by approximating diagonal preconditioning.

---

## 8. Initialization Strategies

Poor initialization can doom training before it begins. Random initialization must satisfy:

1. **Break symmetry**: Different neurons must have different initial parameters
2. **Preserve gradient scale**: Avoid immediate vanishing/exploding gradients

### Xavier (Glorot) Initialization

For layers with sigmoid/tanh activations:
$$W^{(\ell)} \sim \mathcal{N}\left(0, \frac{2}{d_{\ell-1} + d_\ell}\right)$$

or uniformly:
$$W^{(\ell)} \sim \mathcal{U}\left(-\sqrt{\frac{6}{d_{\ell-1} + d_\ell}}, \sqrt{\frac{6}{d_{\ell-1} + d_\ell}}\right)$$

**Motivation**: Preserve variance of activations and gradients across layers.

### He Initialization

For ReLU networks:
$$W^{(\ell)} \sim \mathcal{N}\left(0, \frac{2}{d_{\ell-1}}\right)$$

**Rationale**: Accounts for ReLU killing half the activations on average.

### Biases

Typically initialized to zero: $b^{(\ell)} = 0$, though small positive values (e.g., 0.01) can be used for ReLU to ensure initial activity.

---

## 9. Batch Normalization

A transformative technique that normalizes activations within each mini-batch.

### Forward Pass

For hidden layer activations $z^{(\ell)}$:

$$
\begin{align}
\mu_{\mathcal{B}} &= \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} z_i^{(\ell)} \\
\sigma_{\mathcal{B}}^2 &= \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} (z_i^{(\ell)} - \mu_{\mathcal{B}})^2 \\
\hat{z}_i^{(\ell)} &= \frac{z_i^{(\ell)} - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \\
\tilde{z}_i^{(\ell)} &= \gamma \hat{z}_i^{(\ell)} + \beta
\end{align}
$$

where $\gamma, \beta$ are learnable scale and shift parameters.

### Benefits

- **Reduces internal covariate shift**: Stabilizes distribution of layer inputs
- **Allows higher learning rates**: Less sensitive to initialization
- **Regularization effect**: Noise from batch statistics acts as regularizer
- **Accelerates training**: Often dramatic speedup

### Inference

At test time, use population statistics computed during training:

$$\hat{z}_i^{(\ell)} = \frac{z_i^{(\ell)} - \mu_{pop}}{\sqrt{\sigma_{pop}^2 + \epsilon}}$$

---

## 10. Practical Training Workflow

### 1. Data Preparation
- Split into train/validation/test sets
- Standardize features
- Shuffle training data

### 2. Architecture Design
- Choose depth and width based on problem complexity
- Start simple, add capacity if underfitting

### 3. Initialization
- Use He initialization for ReLU networks
- Xavier for sigmoid/tanh (though ReLU preferred)

### 4. Optimizer Selection
- **Adam**: Good default choice, minimal tuning
- **SGD + Momentum**: Sometimes better generalization, requires more tuning

### 5. Learning Rate
- Start with $\eta = 10^{-3}$ for Adam
- Start with $\eta = 10^{-1}$ to $10^{-2}$ for SGD + Momentum
- Use learning rate finder or validation-based tuning

### 6. Training Loop
```python
for epoch in range(num_epochs):
    for batch in training_data:
        # Forward pass
        predictions = model(batch.x)
        loss = loss_function(predictions, batch.y)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation
    val_loss = evaluate(model, validation_data)
    
    # Learning rate schedule
    scheduler.step(val_loss)
```

### 7. Monitoring
- Track training and validation loss
- Watch for overfitting (divergence of train/val curves)
- Monitor gradient norms for stability

### 8. Regularization
- Implement early stopping based on validation loss
- Add dropout if needed
- Consider weight decay ($L_2$ regularization)

---

## 11. Diagnosing Training Issues

### High Training Loss, High Validation Loss
**Underfitting**: Model lacks capacity or training insufficient.
- Increase model size
- Train longer
- Reduce regularization
- Check for bugs in implementation

### Low Training Loss, High Validation Loss
**Overfitting**: Model memorizes training data.
- Add regularization (dropout, weight decay)
- Get more data
- Data augmentation
- Reduce model complexity
- Early stopping

### Unstable Training (NaN losses)
**Numerical issues**:
- Reduce learning rate
- Check for gradient explosion (apply clipping)
- Verify input normalization
- Check for bugs in loss computation

### Slow Convergence
- Increase learning rate
- Try different optimizer
- Improve initialization
- Add batch normalization
- Check for vanishing gradients

---

## 12. Computational Considerations

### GPU Acceleration

Neural network training is highly parallelizable:
- Matrix multiplications are GPU-friendly
- Mini-batches process independently
- Modern frameworks (PyTorch, TensorFlow) handle GPU execution automatically

**Practical speedup**: 10-100× compared to CPU for large networks.

### Memory Management

Memory usage dominated by:
- **Parameters**: $\Theta(|\theta|)$
- **Activations**: Must cache all $h^{(\ell)}$ for backprop
- **Gradients**: Same size as parameters

**Batch size tradeoff**: Larger batches improve GPU utilization but increase memory.

### Mixed Precision Training

Use FP16 for forward/backward passes, FP32 for parameter updates:
- Reduces memory by ~2×
- Increases throughput on modern GPUs
- Requires loss scaling to prevent underflow

---

## 13. Summary

Backpropagation and optimization form the computational backbone of neural network training:

1. **Backpropagation** efficiently computes gradients via reverse-mode automatic differentiation, making training feasible in high dimensions.

2. **Stochastic gradient descent** and its variants (momentum, Adam) navigate the loss landscape, with adaptivity and stochasticity proving crucial for practical success.

3. **Proper initialization, normalization, and learning rate scheduling** are essential for stable, efficient training.

4. **Modern practices**—batch normalization, adaptive optimizers, learning rate warmup—have made deep network training far more robust than early approaches.

The interplay between these components determines whether training succeeds. Understanding their mathematical foundations and practical implications is essential for effective model development.

---

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV*.
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.

  **Next Lecture**: Weight Imbalances.
