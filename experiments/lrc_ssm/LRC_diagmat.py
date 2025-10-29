'''
Following https://github.com/patrick-kidger/equinox/blob/e50fed9c81605130eede7eb4577db0d71fc2c881/equinox/nn/_rnn.py#L13-108
'''

import math
from typing import Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from equinox._misc import default_floating_dtype
from equinox._module import field, Module
from equinox.nn._misc import default_init


class LRCU(Module, strict=True):
    """A single step of a Liquid Resistance Liquid Capacitance Neuron (LRCU).
    """
    weight_sensory_mu: Array
    weight_sensory_sigma: Array
    weight_sensory_w: Array
    weight_sensory_h: Array

    weight_mu: Array
    weight_sigma: Array
    weight_w: Array
    weight_h: Array

    weight_gleak: Array
    weight_vleak: Array

    weight_elastance_kernel: Array
    weight_elastance_bias: Array
    weight_elastance_shift: Array

    use_symmetric: bool = field(static=True)

    dt: float = field(static=True)

    input_size: int = field(static=True)
    hidden_size: int = field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_symmetric: bool = False, # type of elastance 
        dt: float = 1.0, # time step size
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `dtype`: The dtype to use for all weights and biases in this GRU cell.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending on
            whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        wsmkey, wsskey, wswkey, wshkey, wmkey, wskey, wwkey, whkey, wgkey, wvkey, wekkey, webkey, weskey = jrandom.split(key, 13)
        lim = math.sqrt(1 / hidden_size)

        self.weight_sensory_mu = default_init(wsmkey, (input_size,), dtype, lim)
        self.weight_sensory_sigma = default_init(wsskey, (input_size,), dtype, lim)
        self.weight_sensory_w = default_init(wswkey, (hidden_size, input_size), dtype, lim)
        self.weight_sensory_h = default_init(wshkey, (hidden_size, input_size), dtype, lim)

        self.weight_mu = default_init(wmkey, (hidden_size, ), dtype, lim)
        self.weight_sigma = default_init(wskey, (hidden_size, ), dtype, lim)
        self.weight_w = default_init(wwkey, (hidden_size, ), dtype, lim)
        self.weight_h = default_init(whkey, (hidden_size, ), dtype, lim)

        self.weight_gleak = default_init(wgkey, (hidden_size,), dtype, lim)
        self.weight_vleak = default_init(wvkey, (hidden_size,), dtype, lim)

        self.weight_elastance_kernel = default_init(wekkey, (hidden_size, ), dtype, lim)
        self.weight_elastance_bias = default_init(webkey, (hidden_size,), dtype, lim)
        self.weight_elastance_shift = default_init(weskey, (hidden_size,), dtype, lim)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.use_symmetric = use_symmetric
        self.dt = dt

    def calculate_A(self, hidden: Array, input: Array):
        sensory_syn = jnn.sigmoid(self.weight_sensory_sigma * (input - self.weight_sensory_mu))
        sensory_syn_w = self.weight_sensory_w @ sensory_syn # neural activation

        syn = jnn.sigmoid(self.weight_sigma * (hidden - self.weight_mu))
        syn_w = self.weight_w * syn

        f = self.weight_gleak + sensory_syn_w + syn_w
        a = -jax.nn.sigmoid(f)

        return a 
    
    def calculate_B(self, hidden:Array, input: Array):
        sensory_syn = jnn.sigmoid(self.weight_sensory_sigma * (input - self.weight_sensory_mu))
        sensory_syn_h = self.weight_sensory_h @ sensory_syn # neural activation
        
        syn = jnn.sigmoid(self.weight_sigma * (hidden - self.weight_mu))
        syn_h = self.weight_h * syn

        g = self.weight_gleak + sensory_syn_h + syn_h
        b = self.weight_vleak*jnp.tanh(g)

        return b
    
    @jax.named_scope("eqx.nn.LRCUCell")
    def __call__(
        self, input: Array, hidden: Array, *, key: Optional[PRNGKeyArray] = None
    ):
        """**Arguments:**

        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a JAX array of shape
            `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The updated hidden state, which is a JAX array of shape `(hidden_size,)`.
        """

        A = self.calculate_A(hidden, input)
        B = self.calculate_B(hidden, input)

        # This could be also input dependent, not only state dep.
        elastance_term = self.weight_elastance_kernel * hidden + self.weight_elastance_bias

        if self.use_symmetric: # type of elastance
            elastance = (jnn.sigmoid(elastance_term + self.weight_elastance_shift) - jnn.sigmoid(elastance_term - self.weight_elastance_shift)) * self.dt
        else:
            elastance = jnn.sigmoid(elastance_term) * self.dt

        v_prime = hidden * A + B

        return hidden + elastance * v_prime
    
    