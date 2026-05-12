---
marp: true
theme: default
paginate: true
backgroundColor: #f8f9fa
---

# Introduction to JAX 🚀
A High-Performance Numerical Computing Library
Based on the `learn-jax` Repository

---

# What is JAX?

JAX is Autograd and XLA, brought together for high-performance machine learning research.

- **NumPy Drop-in:** Familiar `jax.numpy` API.
- **Hardware Acceleration:** Native support for CPU, GPU, and TPU.
- **Composable Transformations:** `jit`, `vmap`, `grad`, and more!

---

# Transformation 1: `jax.jit`
**Just-In-Time Compilation**

Compiles your Python code into optimized machine code using XLA (Accelerated Linear Algebra).

```python
import jax.numpy as jnp
from jax import jit

def slow_f(x):
    return jnp.dot(x, x.T)

# Fast, compiled version
fast_f = jit(slow_f)
```
*Benefit: Fuses operations and dramatically reduces execution time.*

---

# Transformation 2: `jax.vmap`
**Automatic Vectorization**

Transforms a function that operates on a single data point into one that operates on batches of data.

```python
from jax import vmap

# Function for a single example
def predict(weights, inputs):
    return jnp.dot(weights, inputs)

# Vectorized to handle batches of inputs
batch_predict = vmap(predict, in_axes=(None, 0))
```
*Benefit: No more complex, slow manual batching loops.*

---

# Transformation 3: `jax.grad`
**Automatic Differentiation**

Computes the gradient of a scalar-valued function automatically.

```python
from jax import grad

def f(x):
    return 3 * x**2 + 2 * x + 1

# df/dx = 6x + 2
df_dx = grad(f)

print(df_dx(2.0)) # Output: 14.0
```
*Benefit: Exact derivatives without manual math.*

---

# Working with Pytrees

JAX treats nested Python structures (lists, tuples, dicts) as "Pytrees".
Use `jax.tree_map` to apply functions across these structures easily.

```python
from jax.tree_util import tree_map

params = {'w': [1.0, 2.0], 'b': 0.5}

# Double everything in the dictionary
doubled = tree_map(lambda x: x * 2, params)
```
*Benefit: Easily update model parameters without writing custom traversal code.*

---

# Summary
JAX gives you the power to write standard Python/NumPy code and make it:
1. **Fast** (`jit`)
2. **Batched** (`vmap`)
3. **Differentiable** (`grad`)

### Next Steps
Try the Lab Manual experiments to get hands-on experience!
