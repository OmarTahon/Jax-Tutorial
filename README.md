# Introduction to JAX: Lab Manual

**Author:** Omar Tahon  
**ID:** 022025001  
**Course:** CSE521: Embedded Machine Learning

Welcome to the **Introduction to JAX** lab manual! This manual is designed to be a self-contained guide based on the foundational concepts of JAX. You will learn how to accelerate your numerical computing and machine learning workflows using JAX's core transformations.

## Table of Contents
1. [What is JAX?](#1-what-is-jax)
2. [Experiment 1: Just-In-Time Compilation (`jit`)](#2-experiment-1-just-in-time-compilation-jit)
3. [Experiment 2: Automatic Vectorization (`vmap`)](#3-experiment-2-automatic-vectorization-vmap)
4. [Experiment 3: Automatic Differentiation (`grad`)](#4-experiment-3-automatic-differentiation-grad)
5. [Experiment 4: Working with Pytrees](#5-experiment-4-working-with-pytrees)

---

## 1. What is JAX?

JAX is a library for high-performance numerical computing and machine learning research. It provides a familiar NumPy-like API (`jax.numpy`) but comes with several superpowers:
- **Hardware Acceleration**: Runs seamlessly on CPU, GPU, and TPU.
- **Function Transformations**: Allows you to compile, vectorize, and differentiate Python functions automatically.

### Installation
If you haven't already, install JAX:
```bash
pip install jax jaxlib
```

---

## 2. Experiment 1: Just-In-Time Compilation (`jit`)

JAX allows you to compile your Python functions into highly optimized machine code using XLA (Accelerated Linear Algebra) via the `jax.jit` transformation.

### The Lab
Let's compare the performance of a standard Python function versus a JIT-compiled function.

```python
import jax
import jax.numpy as jnp
import time

def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# Create a large random array
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000000,))

# 1. Run without JIT
start = time.time()
res1 = selu(x).block_until_ready()
print(f"Without JIT: {time.time() - start:.4f} seconds")

# 2. Run with JIT
selu_jit = jax.jit(selu)
# First run compiles the function (warmup)
_ = selu_jit(x).block_until_ready()

start = time.time()
res2 = selu_jit(x).block_until_ready()
print(f"With JIT: {time.time() - start:.4f} seconds")
```

**Observation:** The JIT-compiled version is significantly faster because XLA fuses the operations together, reducing memory overhead and optimizing execution.

---

## 3. Experiment 2: Automatic Vectorization (`vmap`)

Writing batched code can be error-prone and hard to read. `jax.vmap` allows you to write a function for a single data point and automatically vectorize it to handle batches of data.

### The Lab
Let's compute the dot product of two vectors, and then use `vmap` to compute a matrix-matrix multiplication (batch of dot products).

```python
import jax
import jax.numpy as jnp

# Function for a single 1D dot product
def dot_product(v1, v2):
    return jnp.dot(v1, v2)

# Create two matrices (batches of vectors)
mat1 = jnp.ones((10, 5))
mat2 = jnp.ones((10, 5))

# Vectorize the dot product function
# in_axes=(0, 0) means we map over the first axis (rows) of both inputs
v_dot_product = jax.vmap(dot_product, in_axes=(0, 0))

# Apply to batches
result = v_dot_product(mat1, mat2)
print("Batched result shape:", result.shape) # Expected: (10,)
```

**Observation:** We didn't have to rewrite `dot_product` to handle 2D arrays. `vmap` handled the batching loop internally and efficiently.

---

## 4. Experiment 3: Automatic Differentiation (`grad`)

JAX can automatically compute the gradient (derivative) of a scalar-valued function with respect to its inputs using `jax.grad`.

### The Lab
Let's find the derivative of a simple polynomial function.

```python
import jax
import jax.numpy as jnp

# f(x) = 3x^2 + 2x + 1
# f'(x) = 6x + 2
def f(x):
    return 3 * x**2 + 2 * x + 1

# Create the derivative function
df_dx = jax.grad(f)

x_val = 2.0
print(f"f({x_val}) =", f(x_val))
print(f"f'({x_val}) analytically =", 6*x_val + 2)
print(f"f'({x_val}) via JAX =", df_dx(x_val))
```

**Observation:** `jax.grad` returns exactly the correct derivative without any manual calculus. You can even chain it for higher-order derivatives (`jax.grad(jax.grad(f))`).

---

## 5. Experiment 4: Working with Pytrees

In JAX, a "pytree" is a container of leaf elements and/or more pytrees. Common pytrees include lists, tuples, and dictionaries. JAX provides `jax.tree_map` to apply functions across these nested structures effortlessly.

### The Lab
Let's double all the numbers in a nested dictionary without writing custom recursive loops.

```python
import jax

# A complex nested data structure
params = {
    'weights': [1.0, 2.0, 3.0],
    'bias': {'layer1': 0.5, 'layer2': -0.5}
}

# Double all values in the pytree
doubled_params = jax.tree_util.tree_map(lambda x: x * 2, params)

print("Original:", params)
print("Doubled:", doubled_params)
```

**Observation:** `tree_map` navigates the dictionary and lists automatically, applying the lambda function to every leaf node.

---
## Conclusion
You have now explored the four pillars of JAX: **JIT**, **vmap**, **grad**, and **Pytrees**. Combining these transformations allows you to write clean, mathematical Python code that scales to modern accelerators.