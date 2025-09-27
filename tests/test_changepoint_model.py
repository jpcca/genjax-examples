"""
GenJAX changepoint model example from https://www.gen.dev/tutorials/intro-to-modeling/tutorial
"""

from abc import ABC
from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar

import jax
from jax import jit, vmap

import jax.numpy as jnp


from genjax import (
    Arguments,
    beta,
    flip,
    gen,
    scan,
    normal,
    Trace,
    ChoiceMap,
    gamma,
    Weight,
    GenerativeFunction,
)
from jaxtyping import Array, Float, Bool, PRNGKeyArray

from matplotlib import pyplot as plt
from matplotlib.axes import Axes


@jax.tree_util.register_dataclass
@dataclass
class Interval:
    lower: Float[Array, "..."]
    upper: Float[Array, "..."]


@jax.tree_util.register_dataclass
@dataclass
class Node(ABC):
    idx: int = field(metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclass
class InternalNode(Node):
    left: Node
    right: Node
    interval: Interval


@jax.tree_util.register_dataclass
@dataclass
class LeafNode(Node):
    value: Float[Array, "..."]
    interval: Interval


@jax.tree_util.register_dataclass
@dataclass
class NodeData:
    lower: Array
    upper: Array
    values: Array

    def __getitem__(self, idx: int | slice) -> "NodeData":
        return NodeData(
            lower=self.lower[idx],
            upper=self.upper[idx],
            values=self.values[idx],
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim


@partial(
    jax.tree_util.register_dataclass,
    meta_fields=["node_idx"],
    data_fields=["right_idx", "left_idx"],
)
@dataclass
class Topology:
    LEAF_IDX: ClassVar[int] = -1

    node_idx: Array
    left_idx: Array
    right_idx: Array

    def __getitem__(self, idx: int | slice) -> "Topology":
        return Topology(
            node_idx=self.node_idx,
            left_idx=self.left_idx[idx],
            right_idx=self.right_idx[idx],
        )

    def to_leaf(self, idx: int | slice):
        self.left_idx = self.left_idx.at[idx].set(self.LEAF_IDX)
        self.right_idx = self.right_idx.at[idx].set(self.LEAF_IDX)


@jax.tree_util.register_dataclass
@dataclass
class BinaryTree:
    MAX_NODES: ClassVar[int] = 2 ** (5 + 1) - 1

    topology: Topology
    data: NodeData

    @classmethod
    def from_array(cls, array: Float[Array, "..."]) -> "BinaryTree":
        """initialize maximum complexity tree given MAX_NODES"""

        init_nan = jnp.full(shape=cls.MAX_NODES, fill_value=jnp.nan)
        return cls(
            topology=Topology(
                node_idx=jnp.arange(cls.MAX_NODES, dtype=jnp.int32),
                left_idx=jnp.where(
                    2 * jnp.arange(cls.MAX_NODES, dtype=jnp.int32) + 1 < cls.MAX_NODES,
                    2 * jnp.arange(cls.MAX_NODES, dtype=jnp.int32) + 1,
                    -1,
                ),
                right_idx=jnp.where(
                    2 * jnp.arange(cls.MAX_NODES, dtype=jnp.int32) + 2 < cls.MAX_NODES,
                    2 * jnp.arange(cls.MAX_NODES, dtype=jnp.int32) + 2,
                    -1,
                ),
            ),
            data=NodeData(
                lower=init_nan.at[0].set(jnp.amin(array)),
                upper=init_nan.at[0].set(jnp.amax(array)),
                values=init_nan,
            ),
        )

    @property
    def is_leaf(self) -> Bool[Array, "MAX_NODES"]:
        return (self.topology.left_idx == self.topology.LEAF_IDX) & (
            self.topology.right_idx == self.topology.LEAF_IDX
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __getitem__(self, idx: int | slice) -> "BinaryTree":
        return BinaryTree(
            topology=self.topology[idx],
            data=self.data[idx],
        )


@gen
def leaf(buffer: BinaryTree, idx: int) -> tuple[BinaryTree, int]:
    value = normal(0.0, 1.0) @ f"value_{idx}"
    buffer.data.values = buffer.data.values.at[idx].set(value)

    buffer.topology.to_leaf(idx)
    return buffer, idx


@gen
def branch(buffer: BinaryTree, idx: int) -> tuple[BinaryTree, int]:
    lower, upper = buffer.data.lower[idx], buffer.data.upper[idx]

    frac = beta(2.0, 2.0) @ f"beta_{idx}"
    midpoint = lower + frac * (upper - lower)
    left, right = buffer.topology.left_idx[idx], buffer.topology.right_idx[idx]

    buffer.data.lower = buffer.data.lower.at[left].set(lower)
    buffer.data.upper = buffer.data.upper.at[left].set(midpoint)

    buffer.data.lower = buffer.data.lower.at[right].set(midpoint)
    buffer.data.upper = buffer.data.upper.at[right].set(upper)

    return buffer, idx


@scan(n=BinaryTree.MAX_NODES)  # in breadth-first search order
@gen
def binary_tree(buffer: BinaryTree, idx: int) -> tuple[BinaryTree, int]:
    args = (buffer, idx)

    is_leaf = flip(0.5) @ f"is_leaf_{idx}"
    return (
        leaf.or_else(branch)(buffer.is_leaf[idx] | is_leaf, args, args)
        @ f"leaf_or_else_branch_{idx}"
    )


@jit
@partial(vmap, in_axes=(0, None))
def binary_tree_simulate(key: PRNGKeyArray, xs: Float[Array, "..."]) -> Trace:
    buffer = BinaryTree.from_array(xs)
    return binary_tree.simulate(key, args=(buffer, buffer.topology.node_idx))


def get_values_at(xs: Float[Array, "..."], buffer: BinaryTree) -> Float[Array, "..."]:
    mask = (
        buffer.is_leaf
        & (buffer.data.lower <= jnp.expand_dims(xs, axis=-1))
        & (jnp.expand_dims(xs, axis=-1) <= buffer.data.upper)
    )

    return jnp.matmul(
        mask.astype(xs.dtype), jnp.nan_to_num(buffer.data.values, nan=0.0)
    )


@gen
def changepoint_model(xs: Float[Array, "..."]) -> tuple[BinaryTree, int]:
    buffer = BinaryTree.from_array(xs)
    buffer, idx = binary_tree(buffer, buffer.topology.node_idx) @ "binary_tree"

    noise = gamma(0.5, 0.5) @ "noise"
    normal(get_values_at(xs, buffer), noise) @ "y"

    return buffer, idx


@jit
@partial(vmap, in_axes=(0, None, None, None))
def importance(
    key: PRNGKeyArray, model: GenerativeFunction, constraint: ChoiceMap, args: Arguments
) -> Trace:
    return model.importance(key, constraint, args)


@jit
@partial(vmap, in_axes=(0, None, None, None))
def importance_resampling(
    key: PRNGKeyArray,
    model: GenerativeFunction,
    constraint: ChoiceMap,
    args: Arguments,
    num: int = 100_000,
) -> tuple[BinaryTree, Weight]:
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num)

    trace, weights = importance(keys, model, constraint, args)
    weights -= jax.scipy.special.logsumexp(weights)
    index = jax.random.categorical(key, weights)

    buffer, idx = trace.get_retval()
    return buffer[index], weights[index]


# cpu-side, recursive tree builder
def tree_unflatten(tree_buffer: BinaryTree, j: int, idx: int = 0) -> Node:
    if tree_buffer.is_leaf[j, idx]:
        return LeafNode(
            idx=idx,
            value=tree_buffer.data.values[j, idx],  # type: ignore
            interval=Interval(
                tree_buffer.data.lower[j, idx], tree_buffer.data.upper[j, idx]
            ),
        )
    else:
        return InternalNode(
            idx=idx,
            left=tree_unflatten(
                tree_buffer, j, int(tree_buffer.topology.left_idx[j, idx])
            ),  # type: ignore
            right=tree_unflatten(
                tree_buffer, j, int(tree_buffer.topology.right_idx[j, idx])
            ),  # type: ignore
            interval=Interval(
                tree_buffer.data.lower[j, idx], tree_buffer.data.upper[j, idx]
            ),
        )


def render_node(ax: Axes, node: Node, alpha: float | None = None) -> None:
    match node:
        case LeafNode():
            ax.plot(
                [node.interval.lower, node.interval.upper],
                [node.value, node.value],
                linewidth=5,
                color="k",
                alpha=alpha,
                zorder=-1,
            )
        case InternalNode():
            render_node(ax, node.left, alpha)
            render_node(ax, node.right, alpha)

        case _:
            raise ValueError(f"Unknown node type: {type(node)}")


def render_segments(
    buffer: BinaryTree,
    weights: Float[Array, "..."] = None,
    return_figure: bool = False,
) -> None | tuple[plt.Figure, Axes]:
    N = buffer.shape[0] if buffer.ndim > 1 else 1

    fig, ax = plt.subplots(1, 1)
    plt.title("Changepoint Segments")
    plt.xlabel("x")
    plt.ylabel("y")

    weights = (
        jnp.exp(weights - jax.scipy.special.logsumexp(weights))
        if weights is not None
        else None
    )

    weights = (
        weights.at[weights == jnp.amax(weights)].set(1.0)
        if weights is not None
        else jnp.ones(N) / N
    )

    if buffer.ndim > 1:
        for i in range(N):
            tree = tree_unflatten(buffer, i)
            render_node(ax, tree, alpha=weights[i].item())
    else:
        tree = tree_unflatten(buffer, Ellipsis)
        render_node(ax, tree)

    if return_figure:
        return fig, ax
    else:
        fig.savefig("test_changepoint_model.png")
        plt.close(fig)


def test_binary_tree(n_samples: int = 4, seed: int = 42) -> None:
    xs = jnp.array([0, 1])

    trace: Trace = binary_tree_simulate(
        jax.random.split(jax.random.PRNGKey(seed), n_samples),
        xs,
    )

    buffer, _ = trace.get_retval()
    render_segments(buffer)


def test_changepoint_model_inference(seed: int = 42, noise: float = 0.1) -> None:
    key = jax.random.PRNGKey(seed)
    xs = jnp.linspace(-5, 5, num=50)

    noise = noise * jax.random.normal(key, shape=xs.shape, dtype=xs.dtype)
    ys = (jnp.floor(jnp.abs((xs + 5) / 4)).astype(int) + 1) % 3 + noise

    keys = jax.random.split(key, num=12)
    model = changepoint_model

    constraint = ChoiceMap.kw(y=ys)
    args = (xs,)

    buffer, weights = importance_resampling(keys, model, constraint, args)

    fig, ax = render_segments(buffer, weights, return_figure=True)
    ax.scatter(jnp.expand_dims(xs, axis=0), ys, s=3, color="k")

    fig.savefig("test_changepoint_model.png")
    plt.close(fig)
