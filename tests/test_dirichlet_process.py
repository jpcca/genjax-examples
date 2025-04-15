import jax
import jax.numpy as jnp

import genjax
from genjax import ChoiceMapBuilder as C  # type: ignore
from genjax import categorical, dirichlet, gen, normal  # type: ignore
from genjax._src.core.pytree import Const

key = jax.random.key(0)

# Hyper parameters
PRIOR_VARIANCE = 10.0
OBS_VARIANCE = 1.0
N_DATAPOINTS = 5000
N_CLUSTERS = 40
ALPHA = float(N_DATAPOINTS / (N_CLUSTERS * 10))
PRIOR_MEAN = 50.0
N_ITER = 50

# Debugging mode
DEBUG = True


# Sub generative functions of the bigger model
@gen
def generate_cluster(mean, var):
    cluster_mean = normal(mean, var) @ "mean"
    return cluster_mean


@gen
def generate_cluster_weight(alphas):
    probs = dirichlet(alphas) @ "probs"
    return probs


@gen
def generate_datapoint(probs, clusters):
    idx = categorical(jnp.log(probs)) @ "idx"
    obs = normal(clusters[idx], OBS_VARIANCE) @ "obs"
    return obs


@gen
def generate_datapoints(probs, clusters, n_datapoints):
    idx = categorical(jnp.log(probs), sample_shape=n_datapoints) @ "idx"
    obs = normal(clusters[idx], OBS_VARIANCE) @ "obs"
    return obs


# Main model
@gen
def generate_data(n_clusters: Const[int], n_datapoints: Const[int], alpha: float):  # type: ignore
    clusters = (
        generate_cluster.repeat(n=n_clusters.unwrap())(PRIOR_MEAN, PRIOR_VARIANCE)
        @ "clusters"
    )

    probs = generate_cluster_weight.inline(
        alpha / n_clusters.unwrap() * jnp.ones(n_clusters.unwrap())
    )

    datapoints = generate_datapoints(probs, clusters, n_datapoints) @ "datapoints"

    return datapoints


def infer(datapoints):
    key = jax.random.key(32421)
    args = (Const(N_CLUSTERS), Const(N_DATAPOINTS), ALPHA)
    key, subkey = jax.random.split(key)
    initial_weights = C["probs"].set(jnp.ones(N_CLUSTERS) / N_CLUSTERS)
    constraints = datapoints | initial_weights
    tr, _ = generate_data.importance(subkey, constraints, args)

    if DEBUG:
        all_posterior_means = [tr.get_choices()["clusters", "mean"]]
        all_posterior_weights = [tr.get_choices()["probs"]]
        all_cluster_assignment = [tr.get_choices()["datapoints", "idx"]]

        jax.debug.print("Initial means: {v}", v=all_posterior_means[0])
        jax.debug.print("Initial weights: {v}", v=all_posterior_weights[0])

        for _ in range(N_ITER):
            # Gibbs update on `("clusters", i, "mean")` for each i, in parallel
            key, subkey = jax.random.split(key)
            tr = jax.jit(update_cluster_means)(subkey, tr)
            all_posterior_means.append(tr.get_choices()["clusters", "mean"])

            # # Gibbs update on `("datapoints", i, "idx")` for each `i`, in parallel
            key, subkey = jax.random.split(key)
            tr = jax.jit(update_datapoint_assignment)(subkey, tr)
            all_cluster_assignment.append(tr.get_choices()["datapoints", "idx"])

            # # Gibbs update on `probs`
            key, subkey = jax.random.split(key)
            tr = jax.jit(update_cluster_weights)(subkey, tr)
            all_posterior_weights.append(tr.get_choices()["probs"])

        return all_posterior_means, all_posterior_weights, all_cluster_assignment, tr

    else:
        # One Gibbs sweep consist of updating each latent variable
        def update(carry, _):
            key, tr = carry
            # Gibbs update on `("clusters", i, "mean")` for each i, in parallel
            key, subkey = jax.random.split(key)
            tr = update_cluster_means(subkey, tr)

            # Gibbs update on `("datapoints", i, "idx")` for each `i`, in parallel
            key, subkey = jax.random.split(key)
            tr = update_datapoint_assignment(subkey, tr)

            # Gibbs update on `probs`
            key, subkey = jax.random.split(key)
            tr = update_cluster_weights(subkey, tr)
            return (key, tr), None

        # Overall inference performs a fixed number of Gibbs sweeps
        (key, tr), _ = jax.jit(jax.lax.scan)(update, (key, tr), None, length=N_ITER)
        return tr


def update_cluster_means(key, trace):
    # We can update each cluster in parallel
    # For each cluster, we find the datapoints in that cluster and compute their mean
    datapoint_indexes = trace.get_choices()["datapoints", "idx"]
    datapoints = trace.get_choices()["datapoints", "obs"]
    n_clusters = trace.get_args()[0].unwrap()
    current_means = trace.get_choices()["clusters", "mean"]

    # Count number of points per cluster
    category_counts = jnp.bincount(
        trace.get_choices()["datapoints", "idx"],
        length=n_clusters,
        minlength=n_clusters,
    )

    # Will contain some NaN due to clusters having no datapoint
    cluster_means = (
        jax.vmap(
            lambda i: jnp.sum(jnp.where(datapoint_indexes == i, datapoints, 0)),
            in_axes=(0),
            out_axes=(0),
        )(jnp.arange(n_clusters))
        / category_counts
    )

    # Conjugate update for Normal-iid-Normal distribution
    # See https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
    # Note that there's a typo in the math for the posterior mean.
    posterior_means = (
        PRIOR_VARIANCE
        / (PRIOR_VARIANCE + OBS_VARIANCE / category_counts)
        * cluster_means
        + (OBS_VARIANCE / category_counts)
        / (PRIOR_VARIANCE + OBS_VARIANCE / category_counts)
        * PRIOR_MEAN
    )

    posterior_variances = 1 / (1 / PRIOR_VARIANCE + category_counts / OBS_VARIANCE)

    # Gibbs resampling of cluster means
    key, subkey = jax.random.split(key)
    new_means = (
        generate_cluster.vmap()
        .simulate(key, (posterior_means, posterior_variances))
        .get_choices()["mean"]
    )

    # Remove the sampled Nan due to clusters having no datapoint and pick previous mean in that case, i.e. no Gibbs update for them
    chosen_means = jnp.where(category_counts == 0, current_means, new_means)

    if DEBUG:
        jax.debug.print("Category counts: {v}", v=category_counts)
        jax.debug.print("Current means: {v}", v=cluster_means)
        jax.debug.print("Posterior means: {v}", v=posterior_means)
        jax.debug.print(fmt="Posterior variance: {v}", v=posterior_variances)
        jax.debug.print("Resampled means: {v}", v=new_means)
        jax.debug.print("Chosen means: {v}", v=chosen_means)

    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        subkey, C["clusters", "mean"].set(chosen_means), argdiffs
    )
    return new_trace


def update_datapoint_assignment(key, trace):
    # We want to update the index for each datapoint, in parallel.
    # It means we want to resample the i, but instead of being from the prior
    # P(i | probs), we do it from the local posterior P(i | probs, xs).
    # We need to do it for all addresses ["datapoints", "idx", i],
    # and as these are independent (when conditioned on the rest)
    # we can resample them in parallel.

    # Conjugate update for a categorical is just exact posterior via enumeration
    # P(x | y ) = P(x, y) \ sum_x P(x, y).
    # P(x | y1, y2) = P(x | y1)
    # Sampling from
    # (P(x = 1 | y ), P(x = 2 | y), ...) is the same as
    # sampling from Categorical(P(x = 1, y), P(x = 2, y))
    # as the weights need not be normalized
    # In addition, if the model factorizes as P(x, y1, y2) = P(x, y1)P(y1 | y2),
    # we can further simplify P(y1 | y2) from the categorical as it does not depend on x. More generally We only need to look at the children and parents of x ("idx" in our situation, which are conveniently wrapped in the generate_datapoint generative function).

    def compute_local_density(x, i):
        datapoint_mean = trace.get_choices()["datapoints", "obs", x]
        chm = C["obs"].set(datapoint_mean).at["idx"].set(i)
        clusters = trace.get_choices()["clusters", "mean"]
        probs = trace.get_choices()["probs"]
        args = (probs, clusters)
        model_logpdf, _ = generate_datapoint.assess(chm, args)
        return model_logpdf

    n_clusters = trace.get_args()[0].unwrap()
    n_datapoints = trace.get_args()[1].unwrap()
    local_densities = jax.vmap(
        lambda x: jax.vmap(lambda i: compute_local_density(x, i))(
            jnp.arange(n_clusters)
        )
    )(jnp.arange(n_datapoints))

    # Conjugate update by sampling from posterior categorical
    # Note: I think we could've used something like
    # generate_datapoint.vmap().importance which would perhaps
    # work in a more general setting but would definitely be slower here.
    key, subkey = jax.random.split(key)
    new_datapoint_indexes = genjax.categorical.simulate(
        key, (local_densities,)
    ).get_choices()
    # Gibbs resampling of datapoint assignment to clusters
    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        subkey, C["datapoints", "idx"].set(new_datapoint_indexes), argdiffs
    )
    return new_trace


def update_cluster_weights(key, trace):
    # Count number of points per cluster
    n_clusters = trace.get_args()[0].unwrap()
    category_counts = jnp.bincount(
        trace.get_choices()["datapoints", "idx"],
        length=n_clusters,
        minlength=n_clusters,
    )

    # Conjugate update for Dirichlet distribution
    # See https://en.wikipedia.org/wiki/Dirichlet_distribution#Conjugate_to_categorical_or_multinomial
    new_alpha = ALPHA / n_clusters * jnp.ones(n_clusters) + category_counts

    # Gibbs resampling of cluster weights
    key, subkey = jax.random.split(key)
    new_probs = generate_cluster_weight.simulate(key, (new_alpha,)).get_retval()

    if DEBUG:
        jax.debug.print(fmt="Category counts: {v}", v=category_counts)
        jax.debug.print(fmt="New alpha: {v}", v=new_alpha)
        jax.debug.print(fmt="New probs: {v}", v=new_probs)
    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(subkey, C["probs"].set(new_probs), argdiffs)
    return new_trace


def test_dirichlet_process():

    # Generate synthetic data with N_CLUSTERS clusters evenly spaced
    points_per_cluster = int(N_DATAPOINTS / N_CLUSTERS)
    cluster_indices = jnp.arange(N_CLUSTERS)
    offsets = PRIOR_VARIANCE * (-4 + 8 * cluster_indices / N_CLUSTERS)

    # Create keys for each cluster
    keys = jax.random.split(jax.random.key(0), N_CLUSTERS)

    # Generate uniform random points for each cluster
    uniform_points = jax.vmap(
        lambda k: jax.random.uniform(k, shape=(points_per_cluster,))
    )(keys)

    # Add offset and prior mean to each cluster's points
    shifted_points = uniform_points + (PRIOR_MEAN + offsets[:, None])

    datapoints = C["datapoints", "obs"].set(shifted_points.reshape(-1))

    if DEBUG:
        (
            all_posterior_means,
            all_posterior_weights,
            all_cluster_assignment,
            posterior_trace,
        ) = infer(datapoints)
    else:
        posterior_trace = infer(datapoints)

    posterior_trace
