import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5

class Spring(eqx.Module):
