"""
GenJAX changepoint model example from https://www.gen.dev/tutorials/intro-to-modeling/tutorial
"""

from abc import ABC
from dataclasses import dataclass, field
from functools import partial

import jax
from jax import jit, vmap
import jax.numpy as jnp
from genjax import beta, flip, gen, scan, normal, Trace, gamma
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

    @property
    def is_leaf(self) -> Bool[Array, "MAX_NODES"]:
        return (self.left_idx == -1) & (self.right_idx == -1)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim


def nodebuffer_from(xs: Float[Array, "..."]) -> NodeBuffer:
    init_nan = jnp.full(shape=MAX_NODES, fill_value=jnp.nan)
    return NodeBuffer(
        lower=init_nan.at[0].set(jnp.amin(xs)),
        upper=init_nan.at[0].set(jnp.amax(xs)),
        values=init_nan,
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
    buffer = nodebuffer_from(xs)
    return binary_tree.simulate(key, args=(buffer, buffer.idx))


def get_values_at(xs: Float[Array, "..."], buffer: NodeBuffer) -> Float[Array, "..."]:
    mask = (
        buffer.is_leaf
        & (buffer.lower <= jnp.expand_dims(xs, axis=-1))
        & (jnp.expand_dims(xs, axis=-1) <= buffer.upper)
    )

    return jnp.matmul(mask.astype(xs.dtype), jnp.nan_to_num(buffer.values, nan=0.0))


@gen
def changepoint_model(
    buffer: NodeBuffer, xs: Float[Array, "..."]
) -> tuple[NodeBuffer, int]:
    buffer, idx = binary_tree(buffer, buffer.idx) @ "binary_tree"

    noise = gamma(0.5, 0.5) @ "noise"
    normal(get_values_at(xs, buffer), noise) @ "y"

    return buffer, idx


@jit
@vmap
def changepoint_model_simulate(key: PRNGKeyArray, xs: Float[Array, "..."]) -> Trace:
    return changepoint_model.simulate(key, args=(nodebuffer_from(xs), xs))


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


def render_node(ax: Axes, node: Node, color: str) -> None:
    match node:
        case LeafNode():
            ax.plot(
                [node.interval.lower, node.interval.upper],
                [node.value, node.value],
                linewidth=5,
                color=color,
                zorder=-1,
            )
        case InternalNode():
            render_node(ax, node.left, color)
            render_node(ax, node.right, color)

        case _:
            raise ValueError(f"Unknown node type: {type(node)}")


def render_segments(
    trace: Trace, return_figure: bool = False
) -> None | tuple[plt.Figure, Axes]:
    buffer, idx = trace.get_retval()
    choices = trace.get_choices()
    N = buffer.shape[0] if buffer.ndim > 1 else 1

    fig, ax = plt.subplots(1, 1)
    plt.title("Changepoint Segments")
    plt.xlabel("x")
    plt.ylabel("y")

    rgba_colors = cm.get_cmap("viridis")(jnp.linspace(0, 1, N).tolist())
    colors = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b, a in rgba_colors
    ]

    if buffer.ndim > 1:
        for i in range(N):
            tree = tree_unflatten(buffer, i)
            render_node(ax, tree, colors[i])
    else:
        tree = tree_unflatten(buffer, Ellipsis)
        render_node(ax, tree, colors[0])

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

    render_segments(trace)


def test_changepoint_model_inference(seed: int = 42) -> None:
    xs = jnp.linspace(-5, 5, num=50)

    trace: Trace = changepoint_model.simulate(
        jax.random.PRNGKey(seed),
        args=(nodebuffer_from(xs), xs),
    )

    fig, ax = render_segments(trace, return_figure=True)

    buffer, idx = trace.get_retval()
    choices = trace.get_choices()

    for name in ["y"]:
        _xs = jnp.arange(choices[name].shape[0])
        ax.scatter(xs, choices[name], s=1, label=name)

    ax.legend()
    fig.savefig("test_changepoint_model.png")
    plt.close(fig)
