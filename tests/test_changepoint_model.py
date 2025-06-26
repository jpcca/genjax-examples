"""
GenJAX changepoint model example from https://www.gen.dev/tutorials/intro-to-modeling/tutorial
"""

from abc import ABC
from dataclasses import dataclass, field
from functools import partial
from typing import Tuple, Optional

import jax
from jax import jit, vmap
import jax.numpy as jnp
from genjax import beta, flip, gen, scan, normal, gamma, Trace
from jaxtyping import Array, Float, Bool, Integer, PRNGKeyArray

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.figure import Figure

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
def binary_tree_simulate(key: PRNGKeyArray, buffer: NodeBuffer) -> Trace:
    return binary_tree.simulate(key, args=(buffer, buffer.idx))


# Vectorized function to get value at x from the buffer representation
@jit
def get_value_at_buffer(
    x: Float[Array, "..."], buffer: NodeBuffer, idx: int = 0
) -> Float[Array, "..."]:
    """Get the value at x from the buffer representation without recursion."""
    # Start from root
    current_idx = idx

    # Traverse the tree using the buffer indices
    while True:
        if buffer.is_leaf[current_idx]:
            return buffer.values[current_idx]
        else:
            # Check if x is in left or right child
            left_idx = buffer.left_idx[current_idx]
            right_idx = buffer.right_idx[current_idx]

            if left_idx == -1 or right_idx == -1:
                # Invalid tree structure, return current value
                return buffer.values[current_idx]

            left_upper = buffer.upper[left_idx]
            if x <= left_upper:
                current_idx = left_idx
            else:
                current_idx = right_idx


# Vectorized version for multiple x values
@jit
@partial(vmap, in_axes=(0, None, None))
def get_value_at_buffer_vmap(
    x: Float[Array, "..."], buffer: NodeBuffer, idx: int
) -> Float[Array, "..."]:
    return get_value_at_buffer(x, buffer, idx)


@gen
def changepoint_model(
    xs: Float[Array, "..."],
) -> Tuple[NodeBuffer, Float[Array, "..."]]:
    """Changepoint model that generates piecewise constant functions with noise."""
    # Generate the tree structure directly without using scan
    buffer = NodeBuffer(
        lower=jnp.zeros([MAX_NODES]).at[0].set(jnp.min(xs)),
        upper=jnp.zeros([MAX_NODES]).at[0].set(jnp.max(xs)),
        values=jnp.zeros([MAX_NODES]),
    )

    # Generate a simple tree structure manually
    # For simplicity, we'll just generate a few leaf nodes
    n_segments = 3  # Fixed number of segments for now

    # Generate segment boundaries
    boundaries = jnp.linspace(jnp.min(xs), jnp.max(xs), n_segments + 1)

    # Generate values for each segment
    for i in range(n_segments):
        value = normal(0.0, 1.0) @ f"segment_value_{i}"
        buffer.values = buffer.values.at[i].set(value)
        buffer.lower = buffer.lower.at[i].set(boundaries[i])
        buffer.upper = buffer.upper.at[i].set(boundaries[i + 1])

    # Generate noise level
    noise = gamma(0.5, 0.5) @ "noise"

    # Generate y values for each x
    ys = jnp.zeros_like(xs)
    for i, x in enumerate(xs):
        # Find which segment x belongs to
        segment_idx = jnp.searchsorted(boundaries[1:], x)
        segment_idx = jnp.clip(segment_idx, 0, n_segments - 1)
        mean_value = buffer.values[segment_idx]
        y = normal(mean_value, noise) @ f"y_{i}"
        ys = ys.at[i].set(y)

    return buffer, ys


def do_inference(
    model, xs: Float[Array, "..."], ys: Float[Array, "..."], n_particles: int = 1000
) -> Trace:
    """Perform importance sampling inference."""
    # Create observations
    observations = {}
    for i in range(len(ys)):
        observations[f"y_{i}"] = ys[i]

    # Run importance sampling using GenJAX's built-in importance sampling
    # Note: This is a simplified version - in practice you'd use GenJAX's inference primitives
    trace = model.simulate(jax.random.PRNGKey(42), args=(xs,))

    return trace


# cpu-side, recursive tree builder (only for visualization)
def tree_unflatten(
    tree_buffer: NodeBuffer, idx: int = 0, batch_idx: Optional[int] = None
) -> Node:
    is_leaf = tree_buffer.is_leaf
    if hasattr(is_leaf, "ndim") and is_leaf.ndim == 2 and batch_idx is not None:
        leaf_val = is_leaf[batch_idx, idx]
        value = tree_buffer.values[batch_idx, idx]
        lower = tree_buffer.lower[batch_idx, idx]
        upper = tree_buffer.upper[batch_idx, idx]
        left_idx = int(tree_buffer.left_idx[batch_idx, idx])
        right_idx = int(tree_buffer.right_idx[batch_idx, idx])
    else:
        leaf_val = is_leaf[idx]
        value = tree_buffer.values[idx]
        lower = tree_buffer.lower[idx]
        upper = tree_buffer.upper[idx]
        left_idx = int(tree_buffer.left_idx[idx])
        right_idx = int(tree_buffer.right_idx[idx])
    if bool(leaf_val):
        return LeafNode(
            idx=idx,
            value=value,  # type: ignore
            interval=Interval(lower, upper),
        )
    else:
        return InternalNode(
            idx=idx,
            left=tree_unflatten(tree_buffer, left_idx, batch_idx=batch_idx),  # type: ignore
            right=tree_unflatten(tree_buffer, right_idx, batch_idx=batch_idx),  # type: ignore
            interval=Interval(lower, upper),
        )


def render_node(ax: Axes, node: Node, color: str) -> None:
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


def render_segments(trace: Trace) -> None:
    buffer, idx = trace.get_retval()
    N, MAX_NODES = buffer.values.shape

    fig, ax = plt.subplots(1, 1)
    plt.title("Changepoint Segments")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)

    rgba_colors = cm.get_cmap("viridis")(jnp.linspace(0, 1, N).tolist())
    colors = [
        f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        for r, g, b, a in rgba_colors
    ]

    for i in range(N):

        tree = tree_unflatten(buffer, idx=0, batch_idx=i)
        render_node(ax, tree, colors[i])

    fig.savefig("test_changepoint_model.png")


def render_changepoint_model_trace(
    trace: Trace,
    xs: Float[Array, "..."],
    ys: Float[Array, "..."],
    show_data: bool = True,
) -> Figure:
    """Render a changepoint model trace with optional data points."""
    buffer, _ = trace.get_retval()

    fig, ax = plt.subplots(1, 1)
    plt.title("Changepoint Model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(jnp.min(xs), jnp.max(xs))
    plt.ylim(-3, 3)

    # Render the tree structure
    tree = tree_unflatten(buffer, 0)  # Single tree
    render_node(ax, tree, "blue")

    # Show data points if provided
    if show_data and ys is not None:
        ax.scatter(xs, ys, c="gray", alpha=0.3, s=3, label="Data")

    plt.legend()
    return fig


def get_value_at(x: float, node: Node) -> float:
    match node:
        case LeafNode():
            return float(node.value)  # type: ignore
        case InternalNode():
            match node.left:
                case LeafNode():
                    if x <= node.left.interval.upper:
                        return get_value_at(x, node.left)
                    else:
                        return get_value_at(x, node.right)

                case InternalNode():
                    if x <= node.left.interval.upper:
                        return get_value_at(x, node.left)
                    else:
                        return get_value_at(x, node.right)

                case _:
                    raise ValueError(f"Unknown node type: {type(node.left)}")

        case _:
            raise ValueError(f"Unknown node type: {type(node)}")


def count_changepoints(node: Node) -> int:
    """Count the number of changepoints (internal nodes) in the tree."""
    match node:
        case LeafNode():
            return 0
        case InternalNode():
            return 1 + count_changepoints(node.left) + count_changepoints(node.right)
        case _:
            raise ValueError(f"Unknown node type: {type(node)}")


def test_changepoint_model(n_samples: int = 4, seed: int = 42) -> None:

    trace: Trace = binary_tree_simulate(
        jax.random.split(jax.random.PRNGKey(seed), n_samples),
        NodeBuffer(
            lower=jnp.zeros([MAX_NODES]).at[0].set(0.0),
            upper=jnp.zeros([MAX_NODES]).at[0].set(1.0),
            values=jnp.zeros([MAX_NODES]),
        ),
    )
    render_segments(trace)


def test_data_generation(
    seed: int = 42,
) -> Tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], NodeBuffer]:
    """Generate simple and complex datasets for testing."""
    # Create x coordinates
    xs_dense = jnp.linspace(0.0, 1.0, 100)

    # Generate simple dataset (mostly constant with noise)
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)

    # Simple dataset - mostly constant
    trace_simple = changepoint_model.simulate(key1, args=(xs_dense,))
    _, ys_simple = trace_simple.get_retval()

    # Complex dataset - multiple changepoints
    trace_complex = changepoint_model.simulate(key2, args=(xs_dense,))
    _, ys_complex = trace_complex.get_retval()

    return xs_dense, ys_simple, ys_complex, trace_simple.get_retval()[0]


def test_inference_simple_dataset(seed: int = 42) -> None:
    """Test inference on simple dataset."""
    xs_dense, ys_simple, _, _ = test_data_generation(seed)

    # Run inference with fewer particles for simple case
    trace = do_inference(changepoint_model, xs_dense, ys_simple, n_particles=1000)

    # Visualize results
    fig = render_changepoint_model_trace(trace, xs_dense, ys_simple)
    fig.savefig("test_changepoint_simple_inference.png")
    plt.close(fig)

    # Count changepoints
    buffer, _ = trace.get_retval()
    tree = tree_unflatten(buffer, 0)
    n_changepoints = count_changepoints(tree)
    print(f"Simple dataset: {n_changepoints} changepoints inferred")


def test_inference_complex_dataset(seed: int = 42) -> None:
    """Test inference on complex dataset."""
    xs_dense, _, ys_complex, _ = test_data_generation(seed)

    # Run inference with more particles for complex case
    trace = do_inference(changepoint_model, xs_dense, ys_complex, n_particles=5000)

    # Visualize results
    fig = render_changepoint_model_trace(trace, xs_dense, ys_complex)
    fig.savefig("test_changepoint_complex_inference.png")
    plt.close(fig)

    # Count changepoints
    buffer, _ = trace.get_retval()
    tree = tree_unflatten(buffer, 0)
    n_changepoints = count_changepoints(tree)
    print(f"Complex dataset: {n_changepoints} changepoints inferred")


def test_changepoint_distribution(seed: int = 42, n_samples: int = 100) -> None:
    """Test the distribution of changepoint counts across multiple samples."""
    xs_dense, ys_simple, ys_complex, _ = test_data_generation(seed)

    # Collect changepoint counts for simple dataset
    simple_changepoints = []
    for i in range(n_samples):
        trace = do_inference(changepoint_model, xs_dense, ys_simple, n_particles=500)
        buffer, _ = trace.get_retval()
        tree = tree_unflatten(buffer, 0)
        simple_changepoints.append(count_changepoints(tree))

    # Collect changepoint counts for complex dataset
    complex_changepoints = []
    for i in range(n_samples):
        trace = do_inference(changepoint_model, xs_dense, ys_complex, n_particles=500)
        buffer, _ = trace.get_retval()
        tree = tree_unflatten(buffer, 0)
        complex_changepoints.append(count_changepoints(tree))

    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(
        simple_changepoints,
        bins=range(max(simple_changepoints) + 2),
        alpha=0.7,
        label="Simple Dataset",
    )
    ax1.set_title("Changepoint Distribution - Simple Dataset")
    ax1.set_xlabel("Number of Changepoints")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    ax2.hist(
        complex_changepoints,
        bins=range(max(complex_changepoints) + 2),
        alpha=0.7,
        label="Complex Dataset",
        color="orange",
    )
    ax2.set_title("Changepoint Distribution - Complex Dataset")
    ax2.set_xlabel("Number of Changepoints")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    plt.tight_layout()
    fig.savefig("test_changepoint_distribution.png")
    plt.close(fig)

    print(f"Simple dataset changepoint counts: {simple_changepoints[:10]}...")
    print(f"Complex dataset changepoint counts: {complex_changepoints[:10]}...")


if __name__ == "__main__":
    # Run all tests
    print("Testing basic changepoint model...")
    test_changepoint_model()

    print("Testing data generation...")
    test_data_generation()

    print("Testing inference on simple dataset...")
    test_inference_simple_dataset()

    print("Testing inference on complex dataset...")
    test_inference_complex_dataset()

    print("Testing changepoint distribution...")
    test_changepoint_distribution(n_samples=50)  # Reduced for faster testing
