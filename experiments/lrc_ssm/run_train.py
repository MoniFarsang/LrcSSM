"""
refactor of the code at DEER
https://github.com/machine-discovery/deer/tree/main/experiments/04_rnn_eigenworms
note that use_scan True is sequential evaluation, while False is DEER

refactor and rewrite the code from ELK
https://github.com/lindermanlab/elk/blob/main/experiments/fig3/eigenworms.py
"""

import wandb
import time

import argparse
import os
import sys
from functools import partial
from typing import Tuple, Any, List, Dict, Optional, Sequence, Callable
from glob import glob

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from jax._src import prng

from elk.algs.deer import seq1d
from LRC_diagmat import LRCU as LRCUCell # MF addition
from LRC_diagmat_complex import LRCU as LRCUCellComplex # MF addition

from jaxtyping import Array

import numpy as np

# --------------------------------
#
# model functions
#
# --------------------------------


def vmap_to_shape(func: Callable, shape: Sequence[int]) -> Callable:
    rank = len(shape)
    for i in range(rank - 1):
        func = jax.vmap(func)
    return func


def custom_mlp(
    mlp: eqx.nn.MLP, key: prng.PRNGKeyArray, init_method: Optional[str] = "he_uniform"
) -> eqx.nn.MLP: # from XG
    """
    eqx.nn.MLP with custom initialisation scheme using jax.nn.initializers
    """
    where_bias = lambda m: [lin.bias for lin in m.layers]
    where_weight = lambda m: [lin.weight for lin in m.layers]

    mlp = eqx.tree_at(where=where_bias, pytree=mlp, replace_fn=jnp.zeros_like)

    if init_method is None:
        return mlp

    if init_method == "he_uniform":
        # get all the weights of the mlp model
        weights = where_weight(mlp)
        # split the random key into different subkeys for each layer
        subkeys = jax.random.split(key, len(weights))
        new_weights = [
            jax.nn.initializers.he_uniform()(subkey, weight.shape)
            for weight, subkey in zip(weights, subkeys)
        ]
        mlp = eqx.tree_at(where=where_weight, pytree=mlp, replace=new_weights)
    else:
        return NotImplementedError("only he_uniform is implemented")
    return mlp

def output_mapping(nstate: int, nout: int, dtype: Any, key: prng.PRNGKeyArray) -> Callable:  # MF addition
    """
    Apply the output mapping to the input x using the complex matrix C and vector D.
    The output mapping is defined as (C @ x).real + D.
    """
    C_key, D_key = jax.random.split(key, 2)

    C = jax.random.normal(C_key, shape=(nout, nstate, 2), dtype=dtype)
    D = jax.random.normal(D_key, shape=(nout, ), dtype=dtype)
    # Compute the complex matrix multiplication
    C_complex = C[..., 0] + 1j * C[..., 1]

    output_fn = lambda x: ((C_complex @ x).real + D).astype(dtype)
    return output_fn

class MLP(eqx.Module):
    model: eqx.nn.MLP

    def __init__(self, ninp: int, nstate: int, nout: int, key: prng.PRNGKeyArray, activation: Any = "relu", depth = 1):
        if activation == "relu":
            activation_fn = jax.nn.relu
        elif activation == "tanh":
            activation_fn = jax.nn.tanh
        elif activation == "sigmoid":
            activation_fn = jax.nn.sigmoid
        elif activation == "identity":
            activation_fn = lambda x: x
        self.model = eqx.nn.MLP(
            in_size=ninp,
            out_size=nout,
            width_size=nstate,
            depth=depth, # MF addition
            activation=activation_fn, # MF addition
            key=key,
        )
        self.model = custom_mlp(self.model, key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return vmap_to_shape(self.model, x.shape)(x)

class OutputMapping(eqx.Module): # MF addition
    output_model: Callable

    def __init__(self, nstate: int, nout:int, dtype: Any, key: prng.PRNGKeyArray):
        # create learnable parameters for C and D
        # self.C = np.random.normal(size=(nstate, nstate, 2))
        # self.D = np.random.normal(size=(nstate, ))
        self.output_model = output_mapping(nstate, nout, dtype, key)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Calculate (C@x).real + D and assign it to x
        x = vmap_to_shape(self.output_model, x.shape)(x)
        return x

class GRU(eqx.Module): # This is our gated recurrent unit 
    gru: eqx.Module
    use_scan: bool

    def __init__(self, ninp: int, nstate: int, key: prng.PRNGKeyArray, use_scan: bool, lrc_type: str = "lrc"):
        if lrc_type == "lrc":
            self.gru = LRCUCell(input_size=ninp, hidden_size=nstate, key=key) # MF addition
        elif lrc_type == "lrc_complex":
            self.gru = LRCUCellComplex(input_size=ninp, hidden_size=nstate, key=key) # MF addition
        self.use_scan = use_scan

    def __call__(
        self, inputs: jnp.ndarray, h0: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # h0.shape == (nbatch, nstate)
        # inputs.shape == (nbatch, ninp)
        assert len(inputs.shape) == len(h0.shape)

        states = vmap_to_shape(self.gru, inputs.shape)(inputs, h0)

        return states, states


class SingleScaleGRU(eqx.Module):
    nchannel: int
    nlayer: int
    encoder: MLP
    grus: List[List[GRU]]
    mlps: List[MLP]
    classifier: MLP
    norms: List[eqx.nn.LayerNorm]
    dropout: eqx.nn.Dropout
    dropout_key: prng.PRNGKeyArray
    use_scan: bool
    quasi: bool  # XG addition
    activation: Any # MF addition
    use_norm: bool # MF addition
    complex_state: bool # MF addition
    output_mapping: OutputMapping # MF addition
    ninp_enc: int # MF addition
    ninp: int # MF addition
    nstate: int # MF addition
    dtype: Any = jnp.float32 # MF addition

    def __init__(
        self,
        ninp: int,
        nchannel: int,
        nstate: int,
        nlayer: int,
        nclass: int,
        key: prng.PRNGKeyArray,
        use_scan: bool,
        quasi: bool,
        activation: Any = "relu", # MF addition
        use_norm: bool = True, # MF addition
        lrc_type: str = "lrc", # MF addition
        complex_state: bool = False, # MF addition
        encoder_depth: int = 1, # MF addition
        grupair_depth: int = 1, # MF addition
        classifier_depth: int = 1, # MF addition
        ninp_enc: int = 32, # MF addition
        dtype: Any = jnp.float32, # MF addition
    ):
        keycount = 1 + (nchannel + 1) * nlayer + 1 + 1  # +1 for dropout
        print(f"Keycount: {keycount}")
        keys = jax.random.split(key, keycount)

        self.nchannel = nchannel
        self.nlayer = nlayer
        self.ninp_enc = ninp_enc
        self.ninp = ninp
        self.nstate = nstate

        assert nstate % nchannel == 0
        gru_nstate = int(nstate / nchannel) # leftover from XG, might be worth to think about this. As nchannel is 1, this is just nstate

        self.complex_state = complex_state
        self.dtype = dtype

        self.activation = activation

        # MF encode inputs to a hidden dim - not necessarily equal to nstate
        self.encoder = MLP(ninp=ninp, nstate=ninp_enc, nout=ninp_enc, key=keys[0], depth=encoder_depth) # MF addition

        # nlayers of (scale_gru + mlp) pair
        self.grus = [
            [
                GRU(
                    ninp=ninp_enc, # MF: encoded input dimension (hidden state)
                    nstate=gru_nstate, # MF: this is the ssm dimension
                    key=keys[int(1 + (nchannel * j) + i)],
                    use_scan=use_scan,
                    lrc_type=lrc_type # MF: type of lrc we want to use
                )
                for i in range(nchannel)
            ]
            for j in range(nlayer)
        ]
        self.mlps = [ #MF: output mapping for each gru 
            MLP(
                ninp=nstate, # MF: nstate is the ssm dimension
                nstate=ninp_enc, # MF: this is the num of hidden units in the mlp, might be worth using nstate here too?
                nout=ninp_enc, # MF: this is the output dimension of the mlp, same dim as the input for the next gated unit
                key=keys[int(i + 1 + nchannel * nlayer)],
                activation=self.activation, # MF addition
                depth=grupair_depth, # MF addition
            )
            for i in range(nlayer)
        ]

        if self.complex_state: # MF addition
            self.output_mapping = OutputMapping(nstate, ninp_enc, dtype, keys[-2])
        else:
            self.output_mapping = None

        assert len(self.grus) == nlayer
        assert len(self.grus[0]) == nchannel
        print(
            f"scale_grus random keys end at index {int(1 + (nchannel * (nlayer - 1)) + (nchannel - 1))}"
        )
        print(f"mlps random keys end at index {int((nchannel * nlayer) + nlayer)}")

        # project nstate in the feature dimension to nclasses for classification
        self.classifier = MLP(
            ninp=ninp_enc,
            nstate=ninp_enc,
            nout=nclass,
            key=keys[int((nchannel + 1) * nlayer + 1)],
            depth=classifier_depth
        )

        self.norms = [
            eqx.nn.LayerNorm((ninp_enc,), use_weight=False, use_bias=False)
            for i in range(nlayer * 2)
        ]
        self.dropout = eqx.nn.Dropout(p=0.2)
        self.dropout_key = keys[-1]

        self.use_scan = use_scan
        self.quasi = quasi  # XG addition
        self.use_norm = use_norm

    def __call__(
        self, inputs: jnp.ndarray, h0: jnp.ndarray, yinit_guess: jnp.ndarray
    ) -> jnp.ndarray:
        # encode (or rather, project) the inputs
        inputs = self.encoder(inputs)

        def model_func(carry: jnp.ndarray, inputs: jnp.ndarray, model: Any):
            return model(inputs, carry)[1]  # could be [0] or [1]

        for i in range(self.nlayer):
            if self.use_norm:
                inputs = jax.vmap(self.norms[i])(inputs)  # XG change

            x_from_all_channels = []

            for ch in range(self.nchannel):
                if self.use_scan:
                    model = lambda carry, inputs: self.grus[i][ch](inputs, carry)
                    x = jax.lax.scan(model, h0, inputs)[1]
                    samp_iters = 1
                elif self.quasi:
                    x, samp_iters = seq1d(
                        model_func,
                        h0,
                        inputs,
                        self.grus[i][ch],
                        yinit_guess,
                        quasi=self.quasi,  # XG addition
                        qmem_efficient=False,  # XG addition
                    )
                else:
                    x, samp_iters = seq1d(
                        model_func,
                        h0,
                        inputs,
                        self.grus[i][ch],
                        yinit_guess,
                        quasi=self.quasi,  # XG addition
                    )

                x_from_all_channels.append(x)

            x = jnp.concatenate(x_from_all_channels, axis=-1)

            # MF: map state to the next layer
            if self.complex_state:
                # Calculate (C@x).real + D and assign it to x
                x = self.output_mapping(x)
            elif self.ninp_enc != self.nstate: # MF: if they are not equal, we can't do addition, and better move it here
                x = self.mlps[i](x) #+x # MF: not add +x because it happens in the next lines

            if self.use_norm:
                x = jax.vmap(self.norms[i + 1])(  # XG change
                    x + inputs
                ) # add and norm after multichannel GRU layer
            else:
                x = x + inputs
            
            if self.ninp_enc == self.nstate:
                x = self.mlps[i](x) + x  # XG: add with norm added in the next loop # MF: Moved this before norm if hidden dimension is equal to ssm state dimension
            
            inputs = x

        return self.classifier(x), samp_iters


# --------------------------------
#
# data loading
#
# --------------------------------

from datamodule import DataModule # MF addition


def prep_batch(
    batch: Tuple[torch.Tensor, torch.Tensor], dtype: Any
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert len(batch) == 2
    x, y = batch
    x = jnp.asarray(x.numpy(), dtype=dtype)
    y = jnp.asarray(y.numpy())
    return x, y


def count_params(params) -> jnp.ndarray:
    return sum(
        jnp.prod(jnp.asarray(p.shape)) for p in jax.tree_util.tree_leaves(params)
    )


def grad_norm(grads) -> jnp.ndarray:
    flat_grads = jnp.concatenate(
        [jnp.reshape(g, (-1,)) for g in jax.tree_util.tree_leaves(grads)]
    )
    return jnp.linalg.norm(flat_grads)


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {"loss": loss, "accuracy": accuracy}
    return metrics

def compute_regression_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    mse = jnp.mean((logits - labels) ** 2)
    rmse = jnp.sqrt(mse)
    mae = jnp.mean(jnp.abs(logits - labels))
    metrics = {"mse": mse, "rmse": rmse, "mae": mae}
    return metrics


def get_datamodule(
    batch_size: int, datafile: str = "eigenworms_2345"
) -> pl.LightningDataModule:
    # dset = dset.lower()
    datafile = datafile.lower()
    dset = datafile.split("_")[0]
    if dset in ["eigenworms", "scp1", "scp2", "motor", "heartbeat", "ethanol"]:
        return DataModule(
            batch_size=batch_size, datafile=datafile  # nseq = 17984, nclass = 5
        )
    else:
        return NotImplementedError("Dataset is not available")


# --------------------------------
# code to train rnn
# --------------------------------

# # run on cpu
# jax.config.update('jax_platform_name', 'cpu')
# enable float 64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


@partial(jax.jit, static_argnames=("model"))
def rollout(
    model: eqx.Module,
    y0: jnp.ndarray,
    inputs: jnp.ndarray,
    yinit_guess: List[jnp.ndarray],
) -> jnp.ndarray:
    """
    y0 (nstate,)
    inputs (nsequence, ninp)
    yinit_guess (nsequence, nstate)

    return: (nclass,)
    """
    out, samp_iters = model(inputs, y0, yinit_guess)
    # jax.debug.print(
    #     "inside of rollout, samp_iters is {samp_iters}", samp_iters=samp_iters
    # )
    return out.mean(axis=0), samp_iters


@partial(jax.jit, static_argnames=("static", "classification"))
def loss_fn(
    params: Any,
    static: Any,
    y0: jnp.ndarray,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    yinit_guess: List[jnp.ndarray],
    classification: bool = True,
) -> jnp.ndarray:
    """
    y0 (nbatch, nstate)
    yinit_guess (nbatch, nsequence, nstate)
    batch (nbatch, nsequence, ninp) (nbatch,)
    """
    model = eqx.combine(params, static)
    x, y = batch

    # ypred: (nbatch, nclass)
    ypred, samp_iters = jax.vmap(rollout, in_axes=(None, 0, 0, 0), out_axes=(0))(
        model, y0, x, yinit_guess
    )
    # jax.debug.print(
    #     "inside of loss_fn, samp_iters is {samp_iters}", samp_iters=samp_iters
    # )
    if classification:
        metrics = compute_metrics(ypred, y)
        loss, accuracy = metrics["loss"], metrics["accuracy"]
        return loss, (accuracy, samp_iters)
    else:
        metrics = compute_regression_metrics(ypred, y)
        mse = metrics["mse"]
        return mse, (samp_iters)



@partial(jax.jit, static_argnames=("static", "optimizer", "classification"))
def update_step(
    params: Any,
    static: Any,
    optimizer: optax.GradientTransformation,
    opt_state: Any,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    y0: jnp.ndarray,
    yinit_guess: jnp.ndarray,
    classification: bool = True, # MF addition
) -> Tuple[optax.Params, Any, jnp.ndarray, jnp.ndarray]:
    """
    batch (nbatch, nsequence, ninp) (nbatch,)
    y0 (nbatch, nstate)
    yinit_guess (nbatch, nsequence, nstate)
    """
    loss_and_aux, grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
        params, static, y0, batch, yinit_guess, classification
    )
    if classification:
        loss, (accuracy, samp_iters) = loss_and_aux
    else:
        loss, (samp_iters) = loss_and_aux
        accuracy = 0.0
    updates, opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    gradnorm = grad_norm(grad)
    # jax.debug.print(
    #     "inside of update_step, samp_iters is {samp_iters}", samp_iters=samp_iters
    # )
    return new_params, opt_state, loss, accuracy, gradnorm, samp_iters


def train():
    # set up argparse for the hyperparameters above
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lrc_type", type=str, default="lrc")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--nepochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--ninps", type=int, default=6)
    parser.add_argument("--ninp_enc", type=int, default=32)
    parser.add_argument("--mlp_encoder_depth", type=int, default=1)
    parser.add_argument("--mlp_grupair_depth", type=int, default=1)
    parser.add_argument("--mlp_classifier_depth", type=int, default=1)
    parser.add_argument("--nstates", type=int, default=32)
    parser.add_argument("--nsequence", type=int, default=17984)
    parser.add_argument("--nclass", type=int, default=5)
    parser.add_argument("--nlayer", type=int, default=5)
    parser.add_argument("--nchannel", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--patience_metric", type=str, default="accuracy")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--complex_state", action="store_true", help="Doing --complex_state sets it to True")
    parser.add_argument(
        "--use_scan", action="store_true", help="Doing --use_scan sets it to True"
    )
    parser.add_argument(
        "--quasi", action="store_true", help="Doing --quasi sets it to True"
    )
    parser.add_argument(
        "--datafile",
        type=str,
        default="eigenworms_2345",
    )
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--not_use_norm", action="store_false", help="Doing --not_use_norm sets it to False")
    parser.add_argument("--wandb_user", type=str)
    args = parser.parse_args()

    name = f"{args.datafile}_{args.nlayer}layers_{args.nstates}states_{args.ninp_enc}enc_lr_{args.lr}_mlp_encoder{args.mlp_encoder_depth}_grupair{args.mlp_grupair_depth}_classifier{args.mlp_classifier_depth}_scan_{args.use_scan}_quasi_{args.quasi}_nonlinearity_{args.activation}_norm_{args.not_use_norm}_patience{args.patience}"
    proj_name = f"{args.lrc_type}-ssm"
    wandb.init(project=proj_name, entity=args.wandb_user, name=f"{name}")

    # set seed for pytorch
    torch.manual_seed(42)

    ninp = args.ninps
    nstate = args.nstates
    nsequence = args.nsequence
    nclass = args.nclass
    nlayer = args.nlayer
    nchannel = args.nchannel
    batch_size = args.batch_size
    patience = args.patience
    patience_metric = args.patience_metric
    use_scan = args.use_scan
    quasi = args.quasi  # XG addition
    activation = args.activation
    use_norm = args.not_use_norm
    lrc_type = args.lrc_type

    if args.precision == 32:
        dtype = jnp.float32
    elif args.precision == 64:
        dtype = jnp.float64
    else:
        raise ValueError("Only 32 or 64 accepted")
    
    dtype_init = dtype

    if args.complex_state:
        if args.precision == 32:
            dtype_init = jnp.complex64
        elif args.precision == 64:
            dtype_init = jnp.complex128

    print(f"dtype is {dtype}")
    print(f"use_scan is {use_scan}")
    print(f"quasi is {quasi}")
    print(f"patience_metric is {patience_metric}")

    # check the path
    logpath = "logs_instance"
    path = os.path.join(logpath, f"version_{args.version}")
    # if os.path.exists(path):
    #     raise ValueError(f"Path {path} already exists!")
    os.makedirs(path, exist_ok=True)

    # set up the model and optimizer
    key = jax.random.PRNGKey(args.seed) # for all datasets, use the default seed for the model
    assert nchannel == 1, "currently only support 1 channel" # from XG
    model = SingleScaleGRU(
        ninp=ninp,
        nchannel=nchannel,
        nstate=nstate,
        nlayer=nlayer,
        nclass=nclass,
        key=key,
        use_scan=use_scan,
        quasi=quasi,  # XG addition
        activation=activation, # MF addition
        use_norm=use_norm, # MF addition
        lrc_type=lrc_type, # MF addition
        complex_state=args.complex_state, # MF addition
        encoder_depth=args.mlp_encoder_depth, # MF addition
        grupair_depth=args.mlp_grupair_depth, # MF addition
        classifier_depth=args.mlp_classifier_depth, # MF addition
        ninp_enc=args.ninp_enc, # MF addition
        dtype = dtype, # MF addition
    )
    model = jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if eqx.is_array(x) else x, model
    )
    y0 = jnp.zeros(
        (batch_size, int(nstate / nchannel)), dtype=dtype_init
    )  # (nbatch, nstate)
    yinit_guess = jnp.zeros(
        (batch_size, nsequence, int(nstate / nchannel)), dtype=dtype_init
    )  # (nbatch, nsequence, nstate)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=1), optax.adam(learning_rate=args.lr)
    )
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)
    print(f"Total parameter count: {count_params(params)}")

    # training loop
    step = 0
    dm = get_datamodule(batch_size=args.batch_size, datafile=args.datafile)
    dm.setup()
    classification = True

    best_val_acc = 0
    best_val_loss = float("inf")
    for epoch in tqdm(range(args.nepochs), file=sys.stderr):
        # print(f"starting epoch {epoch}")
        loop = tqdm(
            dm.train_dataloader(),
            total=len(dm.train_dataloader()),
            leave=False,
            file=sys.stderr,
        )
        for i, batch in enumerate(loop):
            try:
                batch = dm.on_before_batch_transfer(batch, i)
            except Exception():
                pass
            batch = prep_batch(batch, dtype)
            t0 = time.time()
            params, opt_state, loss, accuracy, gradnorm, samp_iters = update_step(
                params=params,
                static=static,
                optimizer=optimizer,
                opt_state=opt_state,
                batch=batch,
                y0=y0,
                yinit_guess=yinit_guess,
                classification=classification # MF addition
            )
            t1 = time.time()
            wandb.log(
                {
                    "train_loss": loss,
                    "train_accuracy": accuracy,
                    "gru_gradnorm": gradnorm,
                    "time_per_train_step": t1 - t0,
                    "samp_iters_train": jnp.mean(samp_iters),
                },
                step=step,
            )
            step += 1

        inference_model = eqx.combine(params, static)
        inference_model = eqx.tree_inference(inference_model, value=True)
        inference_params, inference_static = eqx.partition(
            inference_model, eqx.is_array
        )

        if epoch % 1 == 0:
            val_loss = 0
            nval = 0
            val_acc = 0
            loop = tqdm(
                dm.val_dataloader(),
                total=len(dm.val_dataloader()),
                leave=False,
                file=sys.stderr,
            )
            tval = 0
            for i, batch in enumerate(loop):
                try:
                    batch = dm.on_before_batch_transfer(batch, i)
                except Exception():
                    pass
                batch = prep_batch(batch, dtype)
                tstart = time.time()
                if classification:
                    loss, (accuracy, samp_iters) = loss_fn(
                        inference_params, inference_static, y0, batch, yinit_guess, classification
                    )
                else:
                    loss, (samp_iters) = loss_fn(
                        inference_params, inference_static, y0, batch, yinit_guess, classification
                    )
                    accuracy = 0.0
                    
                tval += time.time() - tstart
                val_loss += loss * len(batch[1])
                val_acc += accuracy * len(batch[1])
                nval += len(batch[1])
                # break
            tval /= i + 1
            val_loss /= nval
            val_acc /= nval
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "time_per_val_step": tval,
                    "samp_iters_val": tval,
                },
                step=step,
            )

            if patience_metric == "accuracy":  # MF addition: evaluate test set if validation accuracy is greater than or equal to best_val_acc
                if val_acc >= best_val_acc: 
                    patience = args.patience
                    best_val_acc = val_acc
                    for f in glob(f"{path}/best_model_{name}_epoch_*"):
                        os.remove(f)
                    checkpoint_path = os.path.join(
                        path, f"best_model_{name}_epoch_{epoch}_step_{step}.pkl"
                    )
                    best_model = eqx.combine(params, static)
                    eqx.tree_serialise_leaves(checkpoint_path, best_model)

                    test_loss = 0
                    ntest = 0
                    test_acc = 0
                    loop = tqdm(
                        dm.test_dataloader(),
                        total=len(dm.test_dataloader()),
                        leave=False,
                        file=sys.stderr,
                    )
                    for i, batch in enumerate(loop):
                        try:
                            batch = dm.on_before_batch_transfer(batch, i)
                        except Exception():
                            pass
                        batch = prep_batch(batch, dtype)
                        tstart = time.time()
                        if classification:
                            loss, (accuracy, samp_iters) = loss_fn(
                                inference_params, inference_static, y0, batch, yinit_guess, classification
                            )
                        else:
                            loss, (samp_iters) = loss_fn(
                                inference_params, inference_static, y0, batch, yinit_guess, classification
                            )
                            accuracy = 0.0
                        test_loss += loss * len(batch[1])
                        test_acc += accuracy * len(batch[1])
                        ntest += len(batch[1])
                    test_loss /= ntest
                    test_acc /= ntest
                    wandb.log(
                        {
                            "test_loss": test_loss,
                            "test_accuracy": test_acc,
                        },
                        step=step,
                    )
                else:
                    patience -= 1
                    if patience == 0:
                        print(
                            f"The validation accuracy stopped improving, training ends here at epoch {epoch} and step {step}!"
                        )
                        break
            elif patience_metric == "loss":
                if val_loss <= best_val_loss:
                    patience = args.patience
                    best_val_loss = val_loss
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                    for f in glob(f"{path}/best_model_{name}_epoch_*"):
                        os.remove(f)
                    checkpoint_path = os.path.join(
                        path, f"best_model_{name}_epoch_{epoch}_step_{step}.pkl"
                    )
                    best_model = eqx.combine(params, static)
                    eqx.tree_serialise_leaves(checkpoint_path, best_model)

                    test_loss = 0
                    ntest = 0
                    test_acc = 0
                    loop = tqdm(
                        dm.test_dataloader(),
                        total=len(dm.test_dataloader()),
                        leave=False,
                        file=sys.stderr,
                    )
                    for i, batch in enumerate(loop):
                        try:
                            batch = dm.on_before_batch_transfer(batch, i)
                        except Exception():
                            pass
                        batch = prep_batch(batch, dtype)
                        tstart = time.time()
                        if classification:
                            loss, (accuracy, samp_iters) = loss_fn(
                                inference_params, inference_static, y0, batch, yinit_guess, classification
                            )
                        else:
                            loss, (samp_iters) = loss_fn(
                                inference_params, inference_static, y0, batch, yinit_guess, classification
                            )
                            accuracy = 0.0
                        test_loss += loss * len(batch[1])
                        test_acc += accuracy * len(batch[1])
                        ntest += len(batch[1])
                    test_loss /= ntest
                    test_acc /= ntest
                    wandb.log(
                        {
                            "test_loss": test_loss,
                            "test_accuracy": test_acc,
                        },
                        step=step,
                    )
                else:
                    patience -= 1
                    if patience == 0:
                        print(
                            f"The validation loss stopped improving at {best_val_loss} with accuracy {best_val_acc}, training ends here at epoch {epoch} and step {step}!"
                        )
                        break
            else:
                raise ValueError


if __name__ == "__main__":
    train()
