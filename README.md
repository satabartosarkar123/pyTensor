# pyTensor
A simplified replication of the pytorch library for simple network creation as well as BP based training in miniature dataset

 **A from-scratch implementation of an automatic differentiation engine and a fully functional neural network, built purely in Python.**

This project is an attempt to deeply understand the internals of deep learning frameworks like PyTorch and TensorFlow at an architecture level — by rebuilding the core autograd machinery and neural network abstractions from the ground up.

---

## What This Project Covers

- **`value` class** — A scalar-valued autograd engine that tracks computations and gradients
- **Computation Graph** — Dynamically built DAG of operations, visualized with Graphviz
- **Backpropagation** — Reverse-mode automatic differentiation via topological sort
- **Neural Network** — `Neuron` → `Layer` → `MLP` abstraction with `tanh` activation
- **Training Loop** — MSE loss, gradient descent, and convergence over 20 epochs
- **PyTorch Verification** — Cross-checking custom gradients against PyTorch's autograd

---

## Core Concepts & Math

### 1. The `value` Class — Autograd Engine

Every arithmetic operation wraps its result in a `value` node, forming a **directed acyclic graph (DAG)**. Each node stores:

- `data` — the forward-pass result
- `grad` — the gradient ∂L/∂self, initialized to 0
- `pred` — set of parent nodes (predecessors in the graph)
- `_backprop` — a closure that computes local gradients via the chain rule

### 2. Supported Operations & Their Gradients

**Addition:** `c = a + b`

```
Forward:   c.data = a.data + b.data
Backward:  ∂L/∂a += ∂L/∂c
           ∂L/∂b += ∂L/∂c
```

**Multiplication:** `c = a * b`

```
Forward:   c.data = a.data × b.data
Backward:  ∂L/∂a += b.data × ∂L/∂c
           ∂L/∂b += a.data × ∂L/∂c
```

**Power:** `c = a ** n`

```
Forward:   c.data = a.data ^ n
Backward:  ∂L/∂a += n × a.data^(n−1) × ∂L/∂c
```

**Tanh activation:** `c = tanh(a)`

```
Forward:   c.data = (e^(2a) − 1) / (e^(2a) + 1)
Backward:  ∂L/∂a += (1 − c.data²) × ∂L/∂c
```

**Negation and Subtraction** are derived from multiplication and addition respectively.

> Gradients are **accumulated** (`+=`), not overwritten. This correctly handles nodes used in multiple downstream operations (the multivariate chain rule).

### 3. Backpropagation — Reverse-Mode Autodiff

Backpropagation traverses the computation graph in **reverse topological order**, ensuring every node's gradient is fully accumulated before propagating further back.

```
Algorithm: backprop(root)
───────────────────────────
1. Build topological ordering of all nodes via DFS
2. Set root.grad = 1.0                       ← ∂L/∂L = 1
3. For each node in REVERSED topological order:
     Call node._backprop()                    ← accumulate grads to parents
```

### 4. Neuron, Layer & MLP

**Single Neuron:**

```
o = tanh( Σ(wᵢ · xᵢ) + b )
```

**Layer:** A collection of neurons, each receiving the same input vector.

**MLP:** Sequentially stacked layers — output of layer ℓ becomes input of layer ℓ+1.

**Architecture used:** `MLP(3, [4, 4, 1])` — 3 inputs → 4 neurons → 4 neurons → 1 output

### 5. Training Procedure

**Loss Function:**

```
L = Σᵢ (yᵢ − ŷᵢ)²                          (Mean Squared Error)
```

**Update Rule:**

```
w ← w − η · ∂L/∂w                           (Gradient Descent, η = 0.05)
```

**Each epoch:**

1. **Forward pass** — compute predictions for all inputs
2. **Compute loss** — MSE between predictions and targets
3. **Zero gradients** — reset all `.grad` to 0.0 (prevents accumulation across epochs)
4. **Backward pass** — `loss.backprop()` computes all gradients
5. **Parameter update** — nudge each weight/bias to reduce loss

### 6. PyTorch Cross-Check

The notebook computes the same single neuron (`x1*w1 + x2*w2 + b → tanh`) using `torch.Tensor` with `requires_grad=True`, confirming that the custom engine produces **identical gradients** to PyTorch's autograd.

---

## Training Results

```
Epoch  0  →  loss = 0.3445
Epoch  5  →  loss = 0.0580
Epoch 10  →  loss = 0.0391
Epoch 15  →  loss = 0.0293
Epoch 19  →  loss = 0.0243
```

The network converges smoothly, learning the target mapping `[1, -1, -1, 1]` from 4 training examples.

---

 
