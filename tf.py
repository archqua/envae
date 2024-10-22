from copy import deepcopy
from itertools import chain, combinations
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import tensorflow as tf


def powerset(iterable, lo=1):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(lo, len(s) + 1))


def poe(means, logvars, normalize_outp=True, eps=1e-08):
    taus = tf.exp(-logvars)
    if normalize_outp:
        tau = tf.math.reduce_sum(taus, axis=-1) + 1
        mean = tf.math.reduce_sum(taus * means, axis=-1) / tau
    else:
        tau = tf.math.reduce_sum(taus, axis=-1) + eps
        mean = tf.math.reduce_sum(taus * means, axis=-1) / tau
    return mean, tf.math.log(1 / tau)


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
type ListLike[T] = Union[List[T], Tuple[T]]


class EnVAE(tf.keras.Model):
    """[EnVAE](link) tensorflow implementation."""

    # TODO parameterize N layers to allow linear EnVAE
    def __init__(
        self,
        obs_dim: int,
        lat_dim: int,
        depth: int = 2,
        groups: Union[None, Dict[Any, ListLike[int]], ListLike[int]] = None,
        ngroups: int = 2,
        reg: Union[
            None,
            str,
            Tuple[str, float],
            tf.keras.Regularizer,
        ] = ("l2", 1e-03),
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._lat_dim = lat_dim
        # initialize and check groups
        if isinstance(groups, (list, tuple)):
            if len(groups < 1):
                raise ValueError(
                    f"`groups: list = {groups}` has length {len(groups)}, must be >=1"
                )
            groups = {
                i: tf.constant(groups[i], dtype=tf.int32) for i in range(len(groups))
            }
        self._groups = deepcopy(groups)
        if self._groups is None:
            if obs_dim < ngroups:
                raise ValueError("`groups` is not specified and `ngroups` > `obs_dim`")
            self._groups = {
                n: tf.constant(
                    [i for i in range(obs_dim) if i % ngroups == n],
                    dtype=tf.int32,
                )
                for n in range(ngroups)
            }
        else:
            self._groups = {
                k: tf.constant(v, dtype=tf.int32) for k, v in groups.items()
            }
        gks = list(self._groups.keys())
        for i in range(len(self._groups)):
            gi = self._groups[gks[i]].numpy()
            for j in range(i + 1, len(self._groups)):
                gj = self._groups[gks[j]].numpy()
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
                    tf.keras.layers.InputLayer(shape=(len(self._groups[k]),)),
                ]
                + [
                    tf.keras.layers.Dense(
                        units=2 * lat_dim,
                        kernel_regularizer=getreg(),
                        activation="relu",
                        name=f"{k}_enc_dense_{d}",
                    )
                    for d in range(depth - 1)
                ]
                + [
                    tf.keras.layers.Dense(
                        units=2 * lat_dim,
                        kernel_regularizer=getreg(),
                        name=f"{k}_enc_dense_{depth-1}",
                    ),
                ]
            )
            for k in self._groups
        }
        # initialize decoders
        self.decoders = {
            k: tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(shape=(lat_dim,)),
                ]
                + [
                    tf.keras.layers.Dense(
                        units=lat_dim,
                        kernel_regularizer=getreg(),
                        activation="relu",
                        name=f"{k}_dec_dense_{d}",
                    )
                    for d in range(depth - 1)
                ]
                + [
                    tf.keras.layers.Dense(
                        units=len(self._groups[k]),
                        kernel_regularizer=getreg(),
                        name=f"{k}_dec_dense_{depth-1}",
                    ),
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
                # encoder(X[:, self._groups[g]]),
                # tensorflow moment
                encoder(
                    tf.gather_nd(
                        X,
                        tf.stack(
                            [
                                tf.repeat(
                                    tf.range(X.shape[0])[:, tf.newaxis],
                                    self._groups[0].shape[0],
                                    axis=1,
                                ),
                                tf.repeat(
                                    self._groups[0][tf.newaxis, :],
                                    X.shape[0],
                                    axis=0,
                                ),
                            ],
                            axis=2,
                        ),
                    ),
                ),
                num_or_size_splits=2,
                axis=1,
            )
            means[g] = mean
            logvars[g] = logvar
        if not asdict:
            means = tf.stack(list(means.values()), axis=2)
            logvars = tf.stack(list(logvars.values()), axis=2)
        return means, logvars

    def mopoe(
        self, means: tf.Tensor, logvars: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # means, logvar -> [Batch, Hidden, Groups]
        # discrete uniform sampling -- MoE
        ngroups = means.shape[-1]
        # creating mixtures
        m_means = []
        m_logvars = []
        for experts in powerset(range(means.shape[-1])):
            # tensorflow moment
            req_means = tf.gather_nd(
                means,
                tf.constant(
                    [
                        [[[b, h, g] for g in experts] for h in range(means.shape[1])]
                        for b in range(means.shape[0])
                    ],
                    dtype=tf.int32,
                ),
            )
            req_logvars = tf.gather_nd(
                logvars,
                tf.constant(
                    [
                        [[[b, h, g] for g in experts] for h in range(logvars.shape[1])]
                        for b in range(logvars.shape[0])
                    ],
                    dtype=tf.int32,
                ),
            )
            p_mean, p_logvar = poe(req_means, req_logvars)
            m_means.append(p_mean)
            m_logvars.append(p_logvar)
        m_means = tf.stack(m_means, axis=-1)
        m_logvars = tf.stack(m_logvars, axis=-1)
        return m_means, m_logvars

    def reparameterize(self, means: tf.Tensor, logvars: tf.Tensor) -> tf.Tensor:
        # means, logvars -> [Batch, Hidden, Groups]
        m_means, m_logvars = self.mopoe(means, logvars)
        # m_means, m_logvars -> [Batch, Hidden, Mixture = 2**Groups - 1]
        eps = tf.random.normal(shape=means.shape[:-1])
        # indices are the same across latent space
        # differ only across batch
        indices = tf.cast(
            tf.math.floor(
                tf.random.uniform(m_means.shape[:1], 0.0, float(m_means.shape[-1]))
            ),
            tf.int32,
        )
        mask = (
            indices[:, tf.newaxis, tf.newaxis]
            == tf.range(m_means.shape[-1])[tf.newaxis, tf.newaxis, :]
        )
        mask = tf.broadcast_to(mask, m_means.shape)
        sample_mean = tf.reshape(tf.boolean_mask(m_means, mask), means.shape[:-1])
        sample_delta = eps * tf.exp(
            0.5 * tf.reshape(tf.boolean_mask(m_logvars, mask), logvars.shape[:-1])
        )
        return sample_mean + sample_delta

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        res = tf.empty((z.shape[0], self._obs_dim))
        # TODO
        pass


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
