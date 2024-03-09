import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
import seaborn as sns
import matplotlib.pyplot as plt


t0, t1 = 1, 3
drift = lambda t, y, args: -y
diffusion = lambda t, y, args: 0.1 * t
brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=jr.PRNGKey(0))
terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
solver = Euler()
saveat = SaveAt(dense=True)

sol = diffeqsolve(terms, solver, t0, t1, dt0=0.05, y0=1.0, saveat=saveat)
print(sol.evaluate(1.1))  # DeviceArray(0.89436394)

# Creating a grid of time points
ts = jnp.linspace(t0, t1, 100)

# Evaluating the SDE solution at the grid points
ys = sol.evaluate(ts)

# Plotting the grid points and solution at those points
plt.figure(figsize=(10, 6))
plt.plot(ts, ys, marker='o', markersize=4, label='SDE Solution')
plt.xlabel("Time")
plt.ylabel("Solution")
plt.title("Stochastic Differential Equation Solution at Grid Points")
plt.legend()
plt.show()
