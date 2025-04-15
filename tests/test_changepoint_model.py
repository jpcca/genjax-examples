"""
GenJAX changepoint model example from https://www.gen.dev/tutorials/intro-to-modeling/tutorial
"""

from abc import ABC
from dataclasses import dataclass

import jax
from genjax import beta, flip, gen, normal  # type: ignore[import]
from jaxtyping import Array, Float

MAX_DEPTH = 5
global depth
depth = 0


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


def leaf(interval: Interval) -> LeafNode:
    value = normal(0.0, 1.0) @ "value"
    return LeafNode(value, interval)


def branch(interval: Interval) -> InternalNode:

    fraction = beta(2.0, 2.0) @ "fraction"
    midpoint = interval.lower + fraction * (interval.upper - interval.lower)

    left = generate_segments(interval.lower, midpoint) @ "left"
    right = generate_segments(midpoint, interval.upper) @ "right"

    return InternalNode(left, right, interval)


@gen
def generate_segments(lower: Float[Array, "..."], upper: Float[Array, "..."]) -> Node:
    interval = Interval(lower, upper)
    is_leaf = flip(0.7) @ "is_leaf"
    return jax.lax.cond(is_leaf, leaf, branch, interval)


def test_changepoint_model():
    key = jax.random.PRNGKey(42)
    generate_segments.simulate(key, (0.0, 1.0))
