"""
GenJAX changepoint model example from https://www.gen.dev/tutorials/intro-to-modeling/tutorial
"""

from abc import ABC
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from genjax import beta, flip, gen, normal, Trace  # type: ignore
from jaxtyping import Array, Float, Bool, Integer

from matplotlib import pyplot as plt

MAX_DEPTH = 10
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


@gen
def generate_segments(buffer: TreeBuffer) -> TreeBuffer:

    slot = buffer.ptr - 1
    low = buffer.lower[slot]
    up = buffer.upper[slot]
    depth = buffer.depth[slot]

    args = (buffer, slot, low, up, depth)
    is_leaf = flip(0.7) @ f"is_leaf_{slot}"
    leaf.or_else(branch)(
        is_leaf | (depth >= MAX_DEPTH), args, args
    ) @ f"buffer_idx_{slot}"
    return ()


@gen
def generate_segments_scan(lower: int, upper: int) -> TreeBuffer:

    buffer, _ = jax.lax.scan(
        generate_segments,
        TreeBuffer(
            lower=jnp.zeros([MAX_NODES]).at[0].set(lower),
            upper=jnp.ones([MAX_NODES]).at[0].set(upper),
            depth=jnp.zeros([MAX_NODES], dtype=jnp.int32),
            ptr=1,
            node_lower=jnp.zeros([MAX_NODES]),
            node_upper=jnp.zeros([MAX_NODES]),
            is_leaf=jnp.zeros([MAX_NODES], dtype=bool),
            values=jnp.zeros([MAX_NODES]),
            left_idx=-jnp.ones([MAX_NODES], dtype=jnp.int32),
            right_idx=-jnp.ones([MAX_NODES], dtype=jnp.int32),
            next_node=0,
        ),
        xs=None,
        length=MAX_NODES,
    )
    return buffer


def test_changepoint_model():

    for i in range(10):
        key = jax.random.PRNGKey(i)
        trace: Trace = generate_segments_scan.simulate(key, (0.0, 1.0))

        render_segments_trace(trace)
    plt.savefig("test_changepoint_model.png")


# cpu-side, recursive tree builder
def build_tree(tree_buffer: TreeBuffer, idx: int = 0) -> Node:
    if tree_buffer.is_leaf[idx]:
        return LeafNode(
            value=tree_buffer.values[idx],
            interval=Interval(tree_buffer.node_lower[idx], tree_buffer.node_upper[idx]),
        )
    else:
        return InternalNode(
            left=build_tree(tree_buffer, tree_buffer.left_idx[idx]),
            right=build_tree(tree_buffer, tree_buffer.right_idx[idx]),
            interval=Interval(tree_buffer.node_lower[idx], tree_buffer.node_upper[idx]),
        )


def render_node(node: Node) -> None:
    match node:
        case LeafNode():
            plt.plot(
                [node.interval.lower, node.interval.upper],
                [node.value, node.value],
                linewidth=5,
            )
        case InternalNode():
            render_node(node.left)
            render_node(node.right)

        case _:
            raise ValueError(f"Unknown node type: {type(node)}")


def render_segments_trace(trace: Trace) -> None:
    tree = build_tree(trace.retval)

    plt.figure(figsize=(10, 5))
    plt.title("Changepoint Segments")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(-3, 3)
    render_node(tree)
