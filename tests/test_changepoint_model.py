"""
GenJAX changepoint model example from https://www.gen.dev/tutorials/intro-to-modeling/tutorial
"""

from abc import ABC
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from genjax import beta, flip, gen, scan, normal, Trace  # type: ignore
from jaxtyping import Array, Float, Bool, Integer

from matplotlib import pyplot as plt

MAX_DEPTH = 5
MAX_NODES = 2**MAX_DEPTH - 1


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
    # a “stack” of intervals to process:
    lower: Float[Array, "MAX_NODES"]
    upper: Float[Array, "MAX_NODES"]
    depth: Float[Array, "MAX_NODES"]
    ptr: jnp.integer  # next free slot in the stack

    # buffers for the nodes we build:
    node_lower: Float[Array, "MAX_NODES"]  # shape [MAX_NODES]
    node_upper: Float[Array, "MAX_NODES"]  # shape [MAX_NODES]
    is_leaf: Bool[Array, "MAX_NODES"]  # shape [MAX_NODES], bool
    values: Float[Array, "MAX_NODES"]  # shape [MAX_NODES]    (only for leaves)
    left_idx: Integer[Array, "MAX_NODES"]  # shape [MAX_NODES], int32
    right_idx: Integer[Array, "MAX_NODES"]  # shape [MAX_NODES], int32

    next_node: jnp.integer  # next free index in the node buffers


@gen
def leaf(buffer: TreeBuffer, slot, low, up, depth) -> TreeBuffer:
    idx = buffer.next_node
    buffer.ptr = slot  # consumed one
    buffer.next_node = idx + 1
    buffer.node_lower = buffer.node_lower.at[idx].set(low)
    buffer.node_upper = buffer.node_upper.at[idx].set(up)
    buffer.is_leaf = buffer.is_leaf.at[idx].set(True)

    value = normal(0.0, 1.0) @ f"value_{slot}"
    buffer.values = buffer.values.at[idx].set(value)
    return buffer


@gen
def branch(buffer: TreeBuffer, slot, low, up, depth) -> TreeBuffer:
    idx = buffer.next_node
    frac = beta(2.0, 2.0) @ f"beta_{slot}"
    mid = low + frac * (up - low)
    # push right child then left child onto the stack:
    new_ptr = slot + 2
    sl = buffer.lower
    su = buffer.upper
    sd = buffer.depth
    sl = sl.at[slot].set(low)
    su = su.at[slot].set(mid)
    sd = sd.at[slot].set(depth + 1)
    sl = sl.at[slot + 1].set(mid)
    su = su.at[slot + 1].set(up)
    sd = sd.at[slot + 1].set(depth + 1)

    buffer.ptr = new_ptr
    buffer.next_node = idx + 1
    buffer.node_lower = buffer.node_lower.at[idx].set(low)
    buffer.node_upper = buffer.node_upper.at[idx].set(up)
    buffer.is_leaf = buffer.is_leaf.at[idx].set(False)
    buffer.left_idx = buffer.left_idx.at[idx].set(idx + 1)  # left child will be idx+1
    buffer.right_idx = buffer.right_idx.at[idx].set(
        idx + 1 + (2 ** (MAX_DEPTH - (depth + 1)) - 1)
    )
    buffer.lower = sl
    buffer.upper = su
    buffer.depth = sd
    return buffer


@scan(n=MAX_NODES)
@gen
def generate_segments(
    buffer: TreeBuffer, stack: None = None
) -> tuple[TreeBuffer, None]:

    slot = buffer.ptr - 1
    low = buffer.lower[slot]
    up = buffer.upper[slot]
    depth = buffer.depth[slot]

    args = (buffer, slot, low, up, depth)
    is_leaf = flip(0.7) @ f"is_leaf_{slot}"
    buffer = (
        leaf.or_else(branch)(is_leaf | (depth >= MAX_DEPTH), args, args)
        @ f"leaf_or_else_branch_{slot}"
    )
    return buffer, stack


def test_changepoint_model():

    fig, ax = plt.subplots(1, 1)
    plt.title("Changepoint Segments")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(-3, 3)

    c = ["red", "blue", "green", "orange", "purple", "brown"]
    for i in range(6):
        key = jax.random.PRNGKey(i)
        trace: Trace = generate_segments.simulate(
            key,
            (
                TreeBuffer(
                    lower=jnp.zeros([MAX_NODES]).at[0].set(0.0),
                    upper=jnp.zeros([MAX_NODES]).at[0].set(1.0),
                    depth=jnp.zeros([MAX_NODES], dtype=jnp.int32).at[0].set(0),
                    ptr=1,
                    node_lower=jnp.zeros([MAX_NODES]),
                    node_upper=jnp.zeros([MAX_NODES]),
                    is_leaf=jnp.zeros([MAX_NODES], dtype=bool),
                    values=jnp.zeros([MAX_NODES]),
                    left_idx=-jnp.ones([MAX_NODES], dtype=jnp.int32),
                    right_idx=-jnp.ones([MAX_NODES], dtype=jnp.int32),
                    next_node=0,
                ),
                None,
            ),
        )

        render_segments_trace(ax, trace, color=c[i])
    fig.savefig("test_changepoint_model.png")


# cpu-side, recursive tree builder
def tree_unflatten(tree_buffer: TreeBuffer, idx: int = 0) -> Node:
    if tree_buffer.is_leaf[idx]:
        return LeafNode(
            value=tree_buffer.values[idx],
            interval=Interval(tree_buffer.node_lower[idx], tree_buffer.node_upper[idx]),
        )
    else:
        return InternalNode(
            left=tree_unflatten(tree_buffer, tree_buffer.left_idx[idx]),
            right=tree_unflatten(tree_buffer, tree_buffer.right_idx[idx]),
            interval=Interval(tree_buffer.node_lower[idx], tree_buffer.node_upper[idx]),
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
    buffer, _ = trace.retval
    tree = tree_unflatten(buffer)

    render_node(ax, tree, color)
