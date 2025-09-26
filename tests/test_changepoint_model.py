"""
GenJAX changepoint model example from https://www.gen.dev/tutorials/intro-to-modeling/tutorial
"""

from abc import ABC
from dataclasses import dataclass, field
from functools import partial


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
)
from jaxtyping import Array, Float, Bool, Integer, PRNGKeyArray

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib import cm

MAX_DEPTH = 5
MAX_NODES = 2 ** (MAX_DEPTH + 1) - 1


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
class NodeBuffer:
    lower: Float[Array, "MAX_NODES"]
    upper: Float[Array, "MAX_NODES"]
    values: Float[Array, "MAX_NODES"]

    idx: Integer[Array, "MAX_NODES"] = field(  # in breadth-first search order
        default_factory=lambda: jnp.arange(MAX_NODES, dtype=jnp.int32),
        metadata=dict(static=True),
    )

    left_idx: Integer[Array, "MAX_NODES"] = field(
        default_factory=lambda: jnp.where(
            2 * jnp.arange(MAX_NODES, dtype=jnp.int32) + 1 < MAX_NODES,
            2 * jnp.arange(MAX_NODES, dtype=jnp.int32) + 1,
            -1,
        )
    )

    right_idx: Integer[Array, "MAX_NODES"] = field(
        default_factory=lambda: jnp.where(
            2 * jnp.arange(MAX_NODES, dtype=jnp.int32) + 2 < MAX_NODES,
            2 * jnp.arange(MAX_NODES, dtype=jnp.int32) + 2,
            -1,
        )
    )

    @classmethod
    def from_array(cls, array: Float[Array, "..."]) -> "NodeBuffer":
        init_nan = jnp.full(shape=MAX_NODES, fill_value=jnp.nan)
        return cls(
            lower=init_nan.at[0].set(jnp.amin(array)),
            upper=init_nan.at[0].set(jnp.amax(array)),
            values=init_nan,
        )

    @property
    def is_leaf(self) -> Bool[Array, "MAX_NODES"]:
        return (self.left_idx == -1) & (self.right_idx == -1)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim

    def __getitem__(self, idx: int | slice) -> "NodeBuffer":
        return NodeBuffer(
            lower=self.lower[idx],
            upper=self.upper[idx],
            values=self.values[idx],
            idx=self.idx,
            left_idx=self.left_idx[idx],
            right_idx=self.right_idx[idx],
        )


@gen
def leaf(buffer: NodeBuffer, idx: int) -> tuple[NodeBuffer, int]:
    value = normal(0.0, 1.0) @ f"value_{idx}"
    buffer.values = buffer.values.at[idx].set(value)

    buffer.left_idx = buffer.left_idx.at[idx].set(-1)
    buffer.right_idx = buffer.right_idx.at[idx].set(-1)
    return buffer, idx


@gen
def branch(buffer: NodeBuffer, idx: int) -> tuple[NodeBuffer, int]:
    lower, upper = buffer.lower[idx], buffer.upper[idx]

    frac = beta(2.0, 2.0) @ f"beta_{idx}"
    midpoint = lower + frac * (upper - lower)
    left, right = buffer.left_idx[idx], buffer.right_idx[idx]

    buffer.lower = buffer.lower.at[left].set(lower)
    buffer.upper = buffer.upper.at[left].set(midpoint)

    buffer.lower = buffer.lower.at[right].set(midpoint)
    buffer.upper = buffer.upper.at[right].set(upper)

    return buffer, idx


@scan(n=MAX_NODES)  # in breadth-first search order
@gen
def binary_tree(buffer: NodeBuffer, idx: int) -> tuple[NodeBuffer, int]:
    args = (buffer, idx)

    is_leaf = flip(0.5) @ f"is_leaf_{idx}"
    return (
        leaf.or_else(branch)(buffer.is_leaf[idx] | is_leaf, args, args)
        @ f"leaf_or_else_branch_{idx}"
    )


@jit
@partial(vmap, in_axes=(0, None))
def binary_tree_simulate(key: PRNGKeyArray, xs: Float[Array, "..."]) -> Trace:
    buffer = NodeBuffer.from_array(xs)
    return binary_tree.simulate(key, args=(buffer, buffer.idx))


def get_values_at(xs: Float[Array, "..."], buffer: NodeBuffer) -> Float[Array, "..."]:
    mask = (
        buffer.is_leaf
        & (buffer.lower <= jnp.expand_dims(xs, axis=-1))
        & (jnp.expand_dims(xs, axis=-1) <= buffer.upper)
    )

    return jnp.matmul(mask.astype(xs.dtype), jnp.nan_to_num(buffer.values, nan=0.0))


@gen
def changepoint_model(xs: Float[Array, "..."]) -> tuple[NodeBuffer, int]:
    buffer = NodeBuffer.from_array(xs)
    buffer, idx = binary_tree(buffer, buffer.idx) @ "binary_tree"

    noise = gamma(0.5, 0.5) @ "noise"
    normal(get_values_at(xs, buffer), noise) @ "y"

    return buffer, idx


@jit
@vmap
def changepoint_model_simulate(key: PRNGKeyArray, xs: Float[Array, "..."]) -> Trace:
    return changepoint_model.simulate(key, args=(NodeBuffer.from_array(xs), xs))


# cpu-side, recursive tree builder
def tree_unflatten(tree_buffer: NodeBuffer, j: int, idx: int = 0) -> Node:
    if tree_buffer.is_leaf[j, idx]:
        return LeafNode(
            idx=idx,
            value=tree_buffer.values[j, idx],  # type: ignore
            interval=Interval(tree_buffer.lower[j, idx], tree_buffer.upper[j, idx]),
        )
    else:
        return InternalNode(
            idx=idx,
            left=tree_unflatten(tree_buffer, j, int(tree_buffer.left_idx[j, idx])),  # type: ignore
            right=tree_unflatten(tree_buffer, j, int(tree_buffer.right_idx[j, idx])),  # type: ignore
            interval=Interval(tree_buffer.lower[j, idx], tree_buffer.upper[j, idx]),
        )


def render_node(ax: Axes, node: Node, color: str, weight: float) -> None:
    match node:
        case LeafNode():
            ax.plot(
                [node.interval.lower, node.interval.upper],
                [node.value, node.value],
                linewidth=5,
                color=color,
                alpha=weight,
                zorder=-1,
            )
        case InternalNode():
            render_node(ax, node.left, color, weight)
            render_node(ax, node.right, color, weight)

        case _:
            raise ValueError(f"Unknown node type: {type(node)}")


def render_segments(
    buffer: NodeBuffer,
    return_figure: bool = False,
    weights: Float[Array, "..."] | None = None,
    indeces: Integer[Array, "..."] | None = None,
) -> None | tuple[plt.Figure, Axes]:
    N = buffer.shape[0] if buffer.ndim > 1 else 1

    fig, ax = plt.subplots(1, 1)
    plt.title("Changepoint Segments")
    plt.xlabel("x")
    plt.ylabel("y")

    rgba_colors = cm.get_cmap("viridis")(jnp.linspace(0, 0.1, N).tolist())
    colors = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b, a in rgba_colors
    ]

    # max_weight, min_weight = jnp.percentile(weights, 100), jnp.percentile(weights, 80)
    # weights = (
    #     (weights - min_weight) / (max_weight - min_weight)
    #     if weights is not None
    #     else None
    # )

    weights = 0.5 * jnp.clip(weights, 0, 1.0) if weights is not None else None

    if indeces is None:
        if buffer.ndim > 1:
            for i in range(N):
                tree = tree_unflatten(buffer, i)
                render_node(
                    ax,
                    tree,
                    colors[i],
                    weight=weights[i].item() if weights is not None else 1.0,
                )
        else:
            tree = tree_unflatten(buffer, Ellipsis)
            render_node(
                ax, tree, colors[0], weights.item() if weights is not None else 1.0
            )
    else:
        for i in indeces:
            tree = tree_unflatten(buffer, i)
            render_node(
                ax,
                tree,
                colors[i],
                weight=weights[i].item() if weights is not None else 1.0,
            )
    if return_figure:
        return fig, ax
    else:
        fig.savefig("test_changepoint_model.png")
        plt.close(fig)


def test_changepoint_model(n_samples: int = 4, seed: int = 42) -> None:
    xs = jnp.array([0, 1])

    trace: Trace = binary_tree_simulate(
        jax.random.split(jax.random.PRNGKey(seed), n_samples),
        xs,
    )

    buffer, _ = trace.get_retval()
    render_segments(buffer)


@jit
@partial(vmap, in_axes=(0, None, None))
def changepoint_model_importance(
    key: PRNGKeyArray, constraint: ChoiceMap, args: Arguments
) -> Trace:
    return changepoint_model.importance(key, constraint, args)


def test_changepoint_model_inference(seed: int = 42) -> None:
    key = jax.random.PRNGKey(seed)

    xs = [jnp.linspace(-5, 0, num=50), jnp.linspace(0, 5, num=50)]
    ys = [
        jnp.ones_like(xs[0])
        + 0.1 * jax.random.normal(key, shape=xs[0].shape, dtype=xs[0].dtype),
        -2 * jnp.ones_like(xs[1])
        + 0.5 * jax.random.normal(key, shape=xs[1].shape, dtype=xs[1].dtype),
    ]

    xs = jnp.concatenate(xs)
    ys = jnp.concatenate(ys)

    keys = jax.random.split(key, num=10)
    constraint = ChoiceMap.kw(y=ys)
    args = (xs,)

    buffer, weights = importance_resampling(keys, constraint, args)

    fig, ax = render_segments(buffer, return_figure=True)
    ax.scatter(jnp.expand_dims(xs, axis=0), ys, s=3, label="data", color="gray")

    ax.legend()
    fig.savefig("test_changepoint_model.png")
    plt.close(fig)


@jit
@partial(vmap, in_axes=(0, None, None))
def importance_resampling(
    key: PRNGKeyArray, constraint: ChoiceMap, args: Arguments, num: int = 100000
) -> tuple[NodeBuffer, Weight]:
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num)

    trace, weights = changepoint_model_importance(keys, constraint, args)
    weights = weights - jax.scipy.special.logsumexp(weights)
    index = jax.random.categorical(key, weights)

    buffer, idx = trace.get_retval()
    return buffer[index], weights[index]
