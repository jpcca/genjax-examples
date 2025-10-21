from abc import ABC
from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Bool


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

    def children(self, idx: int | slice) -> tuple[Array, Array]:
        return self.left_idx[idx], self.right_idx[idx]


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
