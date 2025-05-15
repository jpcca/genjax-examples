"""
GenJAX changepoint model example from https://www.gen.dev/tutorials/intro-to-modeling/tutorial
"""

from abc import ABC
from dataclasses import dataclass, field
from functools import partial

import jax
from jax import jit, vmap
import jax.numpy as jnp
from genjax import beta, flip, gen, scan, normal, Trace  # type: ignore
from jaxtyping import Array, Float, Bool, Integer, PRNGKeyArray

from matplotlib import pyplot as plt

MAX_DEPTH = 2
MAX_NODES = 2 ** (MAX_DEPTH + 1) - 1


@jax.tree_util.register_dataclass
@dataclass
class Interval:
    lower: Float[Array, "..."]
    upper: Float[Array, "..."]


@jax.tree_util.register_dataclass
@dataclass
class Node(ABC):
    idx: int


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
class TreeBuffer:

    lower: Float[Array, "MAX_NODES"]
    upper: Float[Array, "MAX_NODES"]
    values: Float[Array, "MAX_NODES"]

    idx: Integer[Array, "MAX_NODES"] = field(  # in breadth-first search order
        default_factory=lambda: jnp.arange(MAX_NODES, dtype=jnp.int32)
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


@gen
def leaf(buffer: TreeBuffer, idx: int) -> tuple[TreeBuffer, int]:

    value = normal(0.0, 1.0) @ f"value_{idx}"
    buffer.values = buffer.values.at[idx].set(value)

    buffer.left_idx = buffer.left_idx.at[idx].set(-1)
    buffer.right_idx = buffer.right_idx.at[idx].set(-1)
    return buffer, idx


def _leaf(buffer: TreeBuffer, idx: int, value: float) -> tuple[TreeBuffer, int]:
    buffer.values = buffer.values.at[idx].set(value)

    buffer.left_idx = buffer.left_idx.at[idx].set(-1)
    buffer.right_idx = buffer.right_idx.at[idx].set(-1)
    return buffer, idx


@gen
def branch(buffer: TreeBuffer, idx: int) -> tuple[TreeBuffer, int]:
    lower, upper = buffer.lower[idx], buffer.upper[idx]

    frac = beta(2.0, 2.0) @ f"beta_{idx}"
    midpoint = lower + frac * (upper - lower)
    left, right = buffer.left_idx[idx], buffer.right_idx[idx]

    buffer.lower = buffer.lower.at[left].set(lower)
    buffer.upper = buffer.upper.at[left].set(midpoint)

    buffer.lower = buffer.lower.at[right].set(midpoint)
    buffer.upper = buffer.upper.at[right].set(upper)

    return buffer, idx


def _branch(buffer: TreeBuffer, idx: int, frac: float) -> tuple[TreeBuffer, int]:
    lower, upper = buffer.lower[idx], buffer.upper[idx]

    midpoint = lower + frac * (upper - lower)
    left, right = buffer.left_idx[idx], buffer.right_idx[idx]

    buffer.lower = buffer.lower.at[left].set(lower)
    buffer.upper = buffer.upper.at[left].set(midpoint)

    buffer.lower = buffer.lower.at[right].set(midpoint)
    buffer.upper = buffer.upper.at[right].set(upper)

    return buffer, idx


@scan(n=MAX_NODES)  # in breadth-first search order
@gen
def binary_tree(buffer: TreeBuffer, idx: int) -> tuple[TreeBuffer, int]:
    args = (buffer, idx)

    frac = beta(2.0, 2.0) @ f"beta_{idx}"
    value = normal(0.0, 1.0) @ f"value_{idx}"
    is_leaf = flip(0.5) @ f"is_leaf_{idx}"

    buffer, idx = jax.lax.cond(
        buffer.is_leaf[idx] | is_leaf,
        lambda buffer, idx: _leaf(buffer, idx, value),
        lambda buffer, idx: _branch(buffer, idx, frac),
        *args,
    )

    # buffer, idx = (
    #     leaf.or_else(branch)(
    #         is_leaf, args, args
    #     )
    #     @ f"leaf_or_else_branch_{idx}"
    # )
    return buffer, idx


@jit
@partial(vmap, in_axes=(0, None))
def sample_binary_tree(key: PRNGKeyArray, buffer: TreeBuffer) -> Trace:
    return binary_tree.simulate(key, args=(buffer, buffer.idx))


def test_changepoint_model():
    colors = ["red", "blue", "green", "orange"]
    N = len(colors)

    trace: Trace = sample_binary_tree(
        jax.random.split(jax.random.PRNGKey(42), N),
        TreeBuffer(
            lower=jnp.zeros([MAX_NODES]).at[0].set(0.0),
            upper=jnp.zeros([MAX_NODES]).at[0].set(1.0),
            values=jnp.zeros([MAX_NODES]),
        ),
    )

    buffer, idx = trace.retval

    fig, ax = plt.subplots(1, 1)
    plt.title("Changepoint Segments")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)

    for i in range(N):

        tree = tree_unflatten(buffer, i)
        render_node(ax, tree, colors[i])

    fig.savefig("test_changepoint_model.png")


# cpu-side, recursive tree builder
def tree_unflatten(tree_buffer: TreeBuffer, j: int, idx: int = 0) -> Node:
    if tree_buffer.is_leaf[j, idx]:
        return LeafNode(
            idx=idx,
            value=tree_buffer.values[j, idx],
            interval=Interval(tree_buffer.lower[j, idx], tree_buffer.upper[j, idx]),
        )
    else:
        return InternalNode(
            idx=idx,
            left=tree_unflatten(tree_buffer, j, tree_buffer.left_idx[j, idx]),
            right=tree_unflatten(tree_buffer, j, tree_buffer.right_idx[j, idx]),
            interval=Interval(tree_buffer.lower[j, idx], tree_buffer.upper[j, idx]),
        )


def render_node(ax: plt.Axes, node: Node, color: str) -> None:
    match node:
        case LeafNode():
            ax.plot(
                [node.interval.lower, node.interval.upper],
                [node.value, node.value],
                linewidth=5,
                color=color,
            )
        case InternalNode():
            render_node(ax, node.left, color)
            render_node(ax, node.right, color)

        case _:
            raise ValueError(f"Unknown node type: {type(node)}")
