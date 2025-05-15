"""
GenJAX changepoint model example from https://www.gen.dev/tutorials/intro-to-modeling/tutorial
"""

from abc import ABC
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from genjax import beta, flip, gen, scan, normal, Trace  # type: ignore
from jaxtyping import Array, Float, Bool, Integer

from matplotlib import pyplot as plt

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
    pass


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

    children: Integer[Array, "MAX_NODES 2"] = field(
        default_factory=lambda: jnp.full((MAX_NODES, 2), -1, dtype=jnp.int32)
    )

    def __post_init__(self):
        idx = jnp.arange(MAX_NODES, dtype=jnp.int32)

        # begin with a full binary tree
        self.children = jnp.stack([2 * idx + 1, 2 * idx + 2], axis=-1)
        self.children = jnp.where(self.children < MAX_NODES, self.children, -1)

    @property
    def is_leaf(self) -> Bool[Array, "MAX_NODES"]:
        return (self.children == -1).all(axis=-1)


@gen
def leaf(buffer: TreeBuffer, idx: int) -> tuple[TreeBuffer, int]:

    value = normal(0.0, 1.0) @ f"value_{idx}"
    buffer.values = buffer.values.at[idx].set(value)

    buffer.children = buffer.children.at[idx].set(-1)
    return buffer, idx


@gen
def branch(buffer: TreeBuffer, idx: int) -> tuple[TreeBuffer, int]:
    lower, upper = buffer.lower[idx], buffer.upper[idx]

    frac = beta(2.0, 2.0) @ f"beta_{idx}"
    midpoint = lower + frac * (upper - lower)

    left, right = buffer.children[idx]

    buffer.lower = buffer.lower.at[left].set(lower)
    buffer.upper = buffer.upper.at[left].set(midpoint)

    buffer.lower = buffer.lower.at[right].set(midpoint)
    buffer.upper = buffer.upper.at[right].set(upper)

    return buffer, idx


@scan(n=MAX_NODES)  # in breadth-first search order
@gen
def binary_tree(buffer: TreeBuffer, idx: int) -> tuple[TreeBuffer, int]:

    frac = beta(2.0, 2.0) @ f"beta_{idx}"
    value = normal(0.0, 1.0) @ f"value_{idx}"

    buffer.values = buffer.values.at[idx].set(value)

    lower, upper = buffer.lower[idx], buffer.upper[idx]
    midpoint = lower + frac * (upper - lower)

    left, right = buffer.children[idx]

    buffer.lower = buffer.lower.at[left].set(lower)
    buffer.upper = buffer.upper.at[left].set(midpoint)

    buffer.lower = buffer.lower.at[right].set(midpoint)
    buffer.upper = buffer.upper.at[right].set(upper)

    # args = (buffer, idx)
    # is_leaf = flip(0.7) @ f"is_leaf_{idx}"
    # buffer, idx = (
    #     leaf.or_else(branch)(
    #         is_leaf | (jnp.floor(jnp.log2(idx + 1)) > MAX_DEPTH), args, args
    #     )
    #     @ f"leaf_or_else_branch_{idx}"
    # )
    return buffer, idx


def test_changepoint_model():

    fig, ax = plt.subplots(1, 1)
    plt.title("Changepoint Segments")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)

    c = ["red", "blue"]
    for i in range(2):

        key = jax.random.PRNGKey(i)
        buffer = TreeBuffer(
            lower=jnp.zeros([MAX_NODES]).at[0].set(0.0),
            upper=jnp.zeros([MAX_NODES]).at[0].set(1.0),
            values=jnp.zeros([MAX_NODES]),
        )
        trace: Trace = binary_tree.simulate(
            key,
            (buffer, buffer.idx),
        )

        render_segments_trace(ax, trace, color=c[i])
    fig.savefig("test_changepoint_model.png")


# cpu-side, recursive tree builder
def tree_unflatten(tree_buffer: TreeBuffer, idx: int = 0) -> Node:
    if tree_buffer.is_leaf[idx]:
        return LeafNode(
            value=tree_buffer.values[idx],
            interval=Interval(tree_buffer.lower[idx], tree_buffer.upper[idx]),
        )
    else:
        return InternalNode(
            left=tree_unflatten(tree_buffer, tree_buffer.children[idx, 0]),
            right=tree_unflatten(tree_buffer, tree_buffer.children[idx, 1]),
            interval=Interval(tree_buffer.lower[idx], tree_buffer.upper[idx]),
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


def render_segments_trace(ax: plt.Axes, trace: Trace, color: str) -> None:
    buffer, idx = trace.retval
    tree = tree_unflatten(buffer)

    render_node(ax, tree, color)
