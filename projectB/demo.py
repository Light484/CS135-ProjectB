# %% [markdown]
# # Autograd Demo
# 
# This file demonstates how to:
# 
# * Use the **autograd** Python package to compute gradients of functions
# * Use gradients from autograd to do a basic linear regression
# 
# # Takeaways
# 
# * Automatic differentiation is a powerful idea that has made experimenting with different models and loss functions far easier than it was even 8 years ago.
# * The Python package `autograd` is a wonderfully simple tool that makes this work with numpy/scipy
# 
# * `autograd` works by a super-smartly implemented version of the backpropagation dynamic programming we've already discussed from Unit 3
#     * Basically, after doing a "forward" pass to evaluate the function, we do a "reverse" pass through the computation graph and compute gradients via the chain rule.
#     * This general purpose method is called [reverse-mode differentiation](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#reverse-mode-differentiation)
# 
# * `autograd` does NOT do symbolic math!
#     * e.g. It does not simplify `ag_np.sqrt(ag_np.square(x))` as `x`. It will use the chain rule on all nested functions that the user specifies.
# * `autograd` does NOT do numerical approximations to gradients.
#     * e.g. It does not estimate gradients by perturbing inputs slightly
# 
# * We'll see how we can define losses in terms of dictionaries, which let us define complicated models with many different parameters. This code specifically is what you'll want to in Project C for matrix factorization with many parameters.
# 
# # Limitations
# 
# FYI There are some things that autograd *cannot* handle that you should be aware of. 
# 
# Make sure any loss function you define that you want to differentiate does not do any of these things:
# 
# * Do not use assignment to elements of arrays, like `A[0] = x` or `A[1] = y`
#     * Instead, compute entries individually and then stack them together.
#     * Like this: `x = ...; y = ...; A = ag_np.hstack([x, y])`
# * Do not rely on implicit casting of lists to arrays, like `A = ag_np.sum([x, y])`
#     * use `A = ag_np.sum(ag_np.array([x, y]))` instead.
# * Do not use A.dot(B) notation
#     * Instead, use `ag_np.dot(A, B)`
# * Avoid in-place operations (such as `a += b`)
#     * Instead, use a = a + b
# 
# # Further Reading
# 
# Check out these great resources
# 
# * Official tutorial for the autograd package: https://github.com/HIPS/autograd/blob/master/docs/tutorial.md
# * Short list of what autograd *can* and *cannot* do: https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#supported-and-unsupported-parts-of-numpyscipy
# 
# 

# %%
## Import numpy
import numpy as np
import pandas as pd
import copy

# %%
## Import autograd
import autograd.numpy as ag_np
import autograd

# %%
# Import plotting libraries
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use('seaborn') # pretty matplotlib plots

import seaborn as sns

# %% [markdown]
# <a name="part1"></a>

# %% [markdown]
# # PART 1: Using autograd.grad for univariate functions
# 
# Suppose we have a mathematical function of interest $f(x)$.
# 
# For now, we'll assume this function has a scalar input and scalar output. This means:
# 
# * $x \in \mathbb{R}$
# * $f(x) \in \mathbb{R}$
# 
# We can ask: what is the derivative (aka *gradient*) of this function:
# 
# $$
# g(x) \triangleq \frac{\partial}{\partial x} f(x)
# $$
# 
# Instead of computing this gradient by hand via calculus/algebra, we can use `autograd` to do it for us.
# 
# First, we need to implement the math function $f(x)$ as a **Python function** `f`.
# 
# The Python function `f` needs to satisfy the following requirements:
# * INPUT 'x': scalar float
# * OUTPUT 'f(x)': scalar float
# * All internal operations are composed of calls to functions from `ag_np`, the `autograd` version of numpy
# 
# ### From numpy to autograd's wrapper of numpy
# 
# You might be used to importing numpy as `import numpy as np`, and then using this shorthand for `np.cos(0.0)` or `np.square(5.0)` etc.
# 
# For autograd to work, you need to instead use **autograd's** provided numpy wrapper interface:
# 
# `from autograd.numpy as ag_np`
# 
# The `ag_np` module has the same API as `numpy`. So for example, you can call
# 
# * `ag_np.cos(0.0)`
# * `ag_np.square(5.0)`
# * `ag_np.sum(a_N)`
# * `ag_np.mean(a_N)`
# * `ag_np.dot(u_NK, v_KM)`
# 
# Or almost any other function you usually would use with `np`
# 
# **Summary:** Make sure your function `f` produces a scalar and only uses functions within the `ag_np` wrapper
# 
# 

# %% [markdown]
# ### Example: f(x) = x^2
# 
# $$
# f(x) = x^2
# $$

# %%
def f(x):
    return ag_np.square(x)

# %%
f(0.0)

# %%
f(1.0)

# %%
f(2.0)

# %% [markdown]
# ### Computing gradients with autograd
# 
# Given a Python function `f` that meets our requirements and evaluates $f(x)$, we want a Python function ``g` that computes the gradient $g(x) \triangleq \frac{\partial}{\partial x}$
# 
# We can use `autograd.grad` to create a Python function `g` 
# 
# ```
# g = autograd.grad(f) # create function g that produces gradients of input function f
# ```
# 
# The symbol `g` is now a **Python function** that takes the same input as `f`, but produces the derivative at a given input.
# 
# 

# %%
g = autograd.grad(f)

# %%
# 'g' is just a function.
# You can call it as usual, by providing a possible scalar float input

g(0.0)

# %%
g(1.0)

# %%
g(2.0)

# %%
g(3.0)

# %% [markdown]
# ### Plot to demonstrate the gradient function  side-by-side with original function

# %%
# Input values evenly spaced between -5 and 5
x_grid_G = np.linspace(-5, 5, 100)

fig_h, subplot_grid = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
subplot_grid[0,0].plot(x_grid_G, [f(x_g) for x_g in x_grid_G], 'k.-')
subplot_grid[0,0].set_title('f(x) = x^2')

subplot_grid[0,1].plot(x_grid_G, [g(x_g) for x_g in x_grid_G], 'b.-')
subplot_grid[0,1].set_title('gradient of f(x)');

# %% [markdown]
# # PART 2: Using autograd.grad for functions with multivariate input
# 

# %% [markdown]
# Now, imagine the input $x$ could be a vector of size D. 
# 
# Our mathematical function $f(x)$ will map each input vector to a scalar.
# 
# We want the gradient function
# 
# \begin{align}
# g(x) &\triangleq \nabla_x f(x)
# \\
# &= [
#     \frac{\partial}{\partial x_1} f(x)
#     \quad \frac{\partial}{\partial x_2} f(x)
#     \quad \ldots \quad \frac{\partial}{\partial x_D} f(x)  ]
# \end{align}
# 
# Instead of computing this gradient by hand via calculus/algebra, we can use autograd to do it for us.
# 
# First, we implement math function $f(x)$ as a **Python function** `f`.
# 
# The Python function `f` needs to satisfy the following requirements:
# * INPUT 'x': numpy array of float
# * OUTPUT 'f(x)': scalar float
# * All internal operations are composed of calls to functions from `ag_np`, the `autograd` version of numpy
# 

# %% [markdown]
# ### Worked Example 2a
# 
# Let's set up a function that is defined as the inner product of the input vector x with some weights $w$
# 
# We assume both $x$ and $w$ are $D$ dimensional vectors
# 
# $$
# f(x) = \sum_{d=1}^D x_d w_d
# $$
# 

# %% [markdown]
# Define the fixed weights

# %%
D = 2

w_D = np.asarray([1., 2.,])

# %% [markdown]
# Define the function `f` using `ag_np` wrapper functions only

# %%
def f(x_D):
    return ag_np.dot(x_D, w_D) # dot product is just inner product in this case

# %% [markdown]
# Use `autograd.grad` to get the gradient function `g`

# %%
g = autograd.grad(f)

# %% [markdown]
# Try putting in the all-zero vector

# %%
x_D = np.zeros(D)

print("x_D", x_D)
print("f(x_D) = %.3f" % (f(x_D)))

# %% [markdown]
# Compute the gradient wrt that all-zero vector

# %%
g(x_D)

# %% [markdown]
# Try another input vector

# %%
x_D = np.asarray([1., 2.])

print("x_D", x_D)
print("f(x_D) = %.3f" % (f(x_D)))

# %% [markdown]
# Compute the gradient wrt the vector [1, 2, 3]

# %%
g(x_D)

# %% [markdown]
# # Part 3: Using autograd gradients within gradient descent to solve multivariate optimization problems

# %% [markdown]
# ### Helper function: basic gradient descent
# 
# Here's a very simple function that will perform many gradient descent steps to optimize a given function.
# 
# 

# %%
def run_many_iters_of_gradient_descent(f, g, init_x_D=None, n_iters=100, step_size=0.001):
    ''' Run many iterations of GD
    
    Args
    ---- 
    f : python function (D,) to float
        Maps vector x_D to scalar loss
    g : python function, (D,) to (D,)
        Maps vector x_D to gradient g_D
    init_x_D : 1D array, shape (D,)
        Initial value for the input vector
    n_iters : int
        Number of gradient descent update steps to perform
    step_size : positive float
        Step size or learning rate for GD
        
    Returns
    -------
    x_D : 1D array, shape (D,)
        Best value of input vector for provided loss f found via this GD procedure
    history : dict
        Contains history of this GD run useful for plotting diagnostics
    '''
    # Copy the initial parameter vector
    x_D = copy.deepcopy(init_x_D)

    # Create data structs to track the per-iteration history of different quantities
    history = dict(
        iter=[],
        f=[],
        x_D=[],
        g_D=[])

    for iter_id in range(n_iters):
        if iter_id > 0:
            x_D = x_D - step_size * g(x_D)

        history['iter'].append(iter_id)
        history['f'].append(f(x_D))
        history['x_D'].append(x_D)
        history['g_D'].append(g(x_D))
    return x_D, history

# %% [markdown]
# ### Worked Example 3a: Minimize f(x) = sum(square(x))
# 
# It's easy to figure out that the vector with smallest L2 norm (smallest sum of squares) is the all-zero vector.
# 
# Here's a quick example of showing that using gradient functions provided by autograd can help us solve the optimization problem:
# 
# $$
# \min_x  \sum_{d=1}^D x_d^2
# $$

# %%
def f(x_D):
    return ag_np.sum(ag_np.square(x_D))

g = autograd.grad(f)

# Initialize at x_D = [6, 4, -3, -5]
D = 4
init_x_D = np.asarray([6.0, 4.0, -3.0, -5.0])

# %%
opt_x_D, history = run_many_iters_of_gradient_descent(f, g, init_x_D, n_iters=1000, step_size=0.01)

# %%
# Make plots of how x parameter values evolve over iterations, and function values evolve over iterations
# Expected result: f goes to zero. all x values goto zero.

fig_h, subplot_grid = plt.subplots(
    nrows=1, ncols=2, sharex=True, sharey=False, figsize=(15,3), squeeze=False)
for d in range(D):
    subplot_grid[0,0].plot(history['iter'], np.vstack(history['x_D'])[:,d], label='x[%d]' % d);
subplot_grid[0,0].set_xlabel('iters')
subplot_grid[0,0].set_ylabel('x_d')
subplot_grid[0,0].legend(loc='upper right')

subplot_grid[0,1].plot(history['iter'], history['f'])
subplot_grid[0,1].set_xlabel('iters')
subplot_grid[0,1].set_ylabel('f(x)');

# %% [markdown]
# # Part 4: Solving linear regression with gradient descent + autograd

# %% [markdown]
# We observe $N$ examples $(x_n, y_n)$ consisting of D-dimensional 'input' vectors $x_n$ and scalar outputs $y_n$.
# 
# Consider the multivariate linear regression model for making a prediction given any input vector $x_i \in \mathbb{R}^D$:
# 
# \begin{align}
# \hat{y}(x_i) = w^T x_i
# \end{align}
# 
# One way to train weights would be to just compute the weights that minimize mean squared error
# 
# \begin{align}
# \min_{w \in \mathbb{R}^D}  \sum_{n=1}^N (y_n - x_n^T w )^2
# \end{align}
# 

# %% [markdown]
# ### Toy Data for linear regression task
# 
# We'll generate data that comes from an idealized linear regression model.
# 
# Each example has D=2 dimensions for x.
# 
# * The first dimension is weighted by +4.2.
# 
# * The second dimension is weighted by -4.2
# 

# %%
N = 100
D = 2
sigma = 0.1

true_w_D = np.asarray([4.2, -4.2])
true_bias = 0.1

train_prng = np.random.RandomState(0)
x_ND = train_prng.uniform(low=-5, high=5, size=(N,D))
y_N = np.dot(x_ND, true_w_D) + true_bias + sigma * train_prng.randn(N)

# %% [markdown]
# ### Toy Data Visualization: Pairplots for all possible (x_d, y) combinations
# 
# You can clearly see the slopes of the lines:
# * x1 vs y plot: slope is around +4
# * x2 vs y plot: slope is around -4

# %%
sns.pairplot(
    data=pd.DataFrame(np.hstack([x_ND, y_N[:,np.newaxis]]), columns=['x1', 'x2', 'y']));

# %%
# Define the optimization problem as an AUTOGRAD-able function wrt the weights w_D
def calc_squared_error_loss(w_D):
    return ag_np.sum(ag_np.square(ag_np.dot(x_ND, w_D) - y_N))

# %%
# Test the *loss function* at the known "ideal" initial point

calc_squared_error_loss(true_w_D)

# %%
# Createa an all-zero weight array to use as our initial guess

init_w_D = np.zeros(2)

# %%
# Test the *loss function* at that all-zero initial point

calc_squared_error_loss(init_w_D)

# %%
# Use autograd.grad to build the gradient function

calc_grad_wrt_w = autograd.grad(calc_squared_error_loss)

# %%
# Test the gradient function at that same initial point 

calc_grad_wrt_w(init_w_D)

# %% [markdown]
# ### Run gradient descent
# 
# Now let's run GD on our simple regression problem

# %%
# Because the gradient's magnitude is very large, use very small step size
opt_w_D, history = run_many_iters_of_gradient_descent(
    calc_squared_error_loss, calc_grad_wrt_w, init_w_D,
    n_iters=400, step_size=0.00001,
    )

# %%
# LinReg worked example
# Make plots of how w_D parameter values evolve over iterations, and function values evolve over iterations
# Expected result: x

fig_h, subplot_grid = plt.subplots(
    nrows=1, ncols=2, sharex=True, sharey=False, figsize=(15,3), squeeze=False)
for d in range(D):
    subplot_grid[0,0].plot(history['iter'], np.vstack(history['x_D'])[:,d], label='w[%d]' % d);
subplot_grid[0,0].set_xlabel('iters')
subplot_grid[0,0].set_ylabel('w_d')
subplot_grid[0,0].legend(loc='upper right')

subplot_grid[0,1].plot(history['iter'], history['f'])
subplot_grid[0,1].set_xlabel('iters')
subplot_grid[0,1].set_ylabel('-1 * log p(y | w, x)');

# %% [markdown]
# # Part 5: Autograd for functions of data structures of arrays

# %% [markdown]
# #### Useful Fact: autograd can take derivatives with respect to DATA STRUCTURES of parameters
# 
# This can help us when it is natural to define models in terms of several parts (e.g. NN layers).
# 
# We don't need to turn our many model parameters into one giant weights-and-biases vector. We can express our thoughts more naturally.

# %% [markdown]
# ### Demo 1: gradient of a LIST of parameters

# %%
def f(w_list_of_arr):
    return ag_np.sum(ag_np.square(w_list_of_arr[0])) + ag_np.sum(ag_np.square(w_list_of_arr[1]))

g = autograd.grad(f)

# %%
w_list_of_arr = [np.zeros(3), np.arange(5, dtype=np.float64)]

print("Type of the gradient is: ")
print(type(g(w_list_of_arr)))

print("Result of the gradient is: ")
g(w_list_of_arr)

# %% [markdown]
# ### Demo 2: gradient of DICT of parameters
# 

# %%
def f(dict_of_arr):
    return ag_np.sum(ag_np.square(dict_of_arr['weights'])) + ag_np.sum(ag_np.square(dict_of_arr['bias']))
g = autograd.grad(f)

# %%
dict_of_arr = dict(weights=np.arange(5, dtype=np.float64), bias=4.2)

print("Type of the gradient is: ")
print(type(g(dict_of_arr)))

print("Result of the gradient is: ")
g(dict_of_arr)


