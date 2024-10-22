from copy import deepcopy
from itertools import chain, combinations
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def poe(means, logvars, normalize_outp=True, eps=1e-08):
    taus = tf.exp(-logvars)
    if normalize_outp:
        tau = taus.sum() + 1
        mean = (taus * means).sum() / tau
    else:
        tau = taus.sum() + eps
        mean = (taus * means).sum() / tau
    return mean, tf.log(1 / tau)


def standard_logp(samples, axis=-1):
    log2pi = tf.math.log(2 * np.pi)
    return tf.reduce_sum(
        -0.5 * (samples**2 + log2pi),
        axis=-1,
    )


def mixture_logp(samples, means, logvars, weights=None):
    # samples -> [Batch, Hidden]
    # means, logvars -> [Batch, Hidden, Mixture]
    samples = samples[:, :, tf.newaxis]
    if weights is None:
        weights = tf.repeat(
            1 / means.shape[-1],
            means.shape[-1],
        )[tf.newaxis, tf.newaxis, :]
    while len(weights.shape) < 3:
        weights = weights[tf.newaxis, ...]
    log2pi = tf.math.log(2.0 * np.pi)
    hisigmas2 = 0.5 * tf.math.exp(-logvars)
    # I hope this is right
    return tf.math.reduce_sum(
        tf.math.log(
            tf.reduce_sum(
                weights
                * tf.sqrt(hisigmas2 / np.pi)
                * tf.math.exp(-hisigmas2 * (samples - means) ** 2),
                axis=2,  # sum over mixture
            )
        ),
        axis=1,  # sum over hidden
    )


T = TypeVar("T")
ListLike[T] = Union[List[T], Tuple[T]]


class EnVAE(tf.keras.Model):
    """[EnVAE](link) tensorflow implementation."""

    # TODO parameterize N layers to allow linear EnVAE
    def __init__(
        self,
        obs_dim: int,
        lat_dim: int,
        reg: Union[
            None,
            str,
            Tuple[str, float],
            tf.keras.Regularizer,
        ] = ("l2", 1e-03),
        groups: Union[None, Dict[Any, ListLike[int]], ListLike[int]] = None,
        ngroups: int = 2,
    ):
        super().__init__()
        # initialize and check groups
        if isinstance(groups, (list, tuple)):
            if len(groups < 1):
                raise ValueError(
                    f"`groups: list = {groups}` has length {len(groups)}, must be >=1"
                )
            groups = {i: groups[i] for i in range(len(groups))}
        self._groups = deepcopy(groups)
        if self._groups is None:
            if obs_dim < ngroups:
                raise ValueError("`groups` is not specified and `ngroups` > `obs_dim`")
            self._groups = {
                n: [i for i in range(obs_dim) if i % ngroups == n]
                for n in range(ngroups)
            }
        gks = list(self._groups.keys())
        for i in range(len(self._groups)):
            gi = self._groups[gks[i]]
            for j in range(i, len(self._groups)):
                gj = self._groups[gks[j]]
                if len(set(gi) & set(gj)) > 0:
                    raise ValueError(
                        f"`groups` has intersecting groups: {gi} & {gj} == {set(gi) & set(gj)}"
                    )

        # regularization factory
        def getreg() -> Optional[tf.keras.Regularizer]:
            if reg is None:
                return None
            if isinstance(reg, tf.keras.Regularizer):
                return deepcopy(reg)
            if isinstance(reg, str):
                if reg.lower() == "l2":
                    return tf.keras.regularizers.L2()
                elif reg.lower() == "l1":
                    return tf.keras.regularizers.L1()
                else:
                    return ValueError(
                        f"`reg: str = {reg}`, must be either 'l2' or 'l1'"
                    )
            elif isinstance(reg, tuple):
                if len(reg) != 2:
                    raise TypeError(
                        f"`reg: tuple = {reg}` has length {len(reg)}, must have length 2"
                    )
                if not isinstance(reg[0], str):
                    raise TypeError(
                        f"`reg[0] = {reg[0]}` is of type {type(reg[0])}, must be of type `str`"
                    )
                if not isinstance(reg[1], float):
                    raise TypeError(
                        f"`reg[1] = {reg[1]}` is of type {type(reg[1])}, must be of type `float`"
                    )
                if reg[0].lower() == "l2":
                    return tf.keras.regularizers.L2(reg[1])
                elif reg[0].lower() == "l1":
                    return tf.keras.regularizers.L1(reg[1])
                else:
                    return ValueError(
                        f"reg[0]: str = {reg[0]}, must be either 'l2' or 'l1'"
                    )
            else:
                raise TypeError(
                    f"`reg` must be `None`, `str` or `tuple`, not {type(reg)}"
                )

        # TODO use functional api to specify inputs and outputs everywhere
        # initialize encoders
        self.encoders = {
            k: tf.keras.Sequential(
                [
                    tf.keras.InputLayer(input_shape=(len(self._groups[k]),)),
                    tf.keras.Dense(units=2 * lat_dim, kernel_regularizer=getreg()),
                    tf.keras.ReLU(),
                    tf.keras.Dense(units=2 * lat_dim, kernel_regularizer=getreg()),
                ]
            )
            for k in self._groups
        }
        # initialize decoders
        self.decoders = {
            k: tf.keras.Sequential(
                [
                    tf.keras.InputLayer(input_shape=(lat_dim,)),
                    tf.keras.Dense(units=lat_dim),
                    tf.keras.ReLU(),
                    tf.keras.Dense(units=len(self._groups[k])),
                ]
            )
            for k in self._groups
        }

        @property
        def groups(self) -> Dict[Any, ListLike[int]]:
            return deepcopy(self._groups)

        @tf.function
        def sample(self, epsilon: tf.Tensor) -> tf.Tensor:
            return self.decode(epsilon)

        def encode(self, X: tf.Tensor, asdict=False) -> tf.Tensor:
            means = dict()
            logvars = dict()
            for g, encoder in self.encoders.items():
                mean, logvar = tf.split(
                    encoder(X[:, self._groups[g]]),
                    num_or_size_splits=2,
                    axis=1,
                )
                means[g] = mean
                logvars[g] = logvar
            if asdict:
                return means, logvars
            else:
                return tf.stack(means.values()), tf.stack(logvars.values())

        def reparameterize(self, means: tf.Tensor, logvars: tf.Tensor) -> tf.Tensor:
            # means, logvar -> [Batch, Hidden, Mixture]
            eps = tf.random.normal(shape=means.shape[:-1])
            indices = tf.cast(
                tf.math.floor(
                    tf.random.uniform(means.shape[:-1], 0.0, float(means.shape[-1]))
                ),
                tf.int32,
            )
            mask = (
                indices[:, :, tf.newaxis]
                == tf.range(means.shape[:-1])[tf.newaxis, tf.newaxis, :]
            )
            return tf.boolean_mask(means, mask) + eps * tf.exp(
                0.5 * tf.boolean_mask(logvars, mask)
            )


def mixture_klde_fn(envae: EnVAE):

    def mixture_klde(means: tf.Tensor, logvars: tf.Tensor, kind: str = "mc"):
        kind = kind.lower()
        if kind == "mc":
            z = envae.reparameterize(means, logvars)

            def wrapped():
                logpz = standard_logp(z)
                logpz_x = mixture_logp(z, means, logvars)
                return -tf.reduce_mean(logpz - logpz_x)

            return wrapped
        else:
            raise NotImplementedError(
                f"{kind} KL divergence estimation is not implemented"
            )

    return mixture_klde_fn
