import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

from genjax import gen, beta, bernoulli, categorical, scan, Trace, ChoiceMap, Diff  # type: ignore
from genjax import ChoiceMapBuilder as C
from jaxtyping import Array, Float, Integer, PRNGKeyArray
from dataclasses import dataclass, field
import numpy as np


# --- SBP module---
@jax.tree_util.register_dataclass
@dataclass
class StickBreakingResult:
    weights: Array


@jax.tree_util.register_dataclass
@dataclass
class _SBPLoopState:
    weights: Array
    remaining_stick: Float[Array, ""]
    alpha: Float[Array, ""]


@gen
def _sbp_step(state: _SBPLoopState, k: int) -> tuple[_SBPLoopState, Float[Array, ""]]:
    v_draw = beta(1.0, state.alpha) @ f"v_draw_for_step_{k}"
    current_weight_value = v_draw * state.remaining_stick
    updated_weights = state.weights.at[k].set(current_weight_value)
    new_remaining_stick = state.remaining_stick * (1.0 - v_draw)
    next_state = _SBPLoopState(
        weights=updated_weights, remaining_stick=new_remaining_stick, alpha=state.alpha
    )
    return next_state, current_weight_value


def create_sbp_model(num_max_components: int):
    if not isinstance(num_max_components, int):
        raise TypeError(
            f"create_sbp_model expects a Python int for num_max_components, but got {type(num_max_components)} with value {num_max_components}"
        )
    if num_max_components <= 1:
        raise ValueError(
            f"num_max_components must be > 1 for SBP, got {num_max_components}"
        )
    scanned_body = scan(n=num_max_components - 1)(_sbp_step)

    @gen
    def sbp_instance_model(alpha_param: float) -> StickBreakingResult:
        initial_state = _SBPLoopState(
            weights=jnp.zeros(num_max_components, dtype=jnp.float32),
            remaining_stick=jnp.array(1.0, dtype=jnp.float32),
            alpha=jnp.array(alpha_param, dtype=jnp.float32),
        )
        indices_for_scan = jnp.arange(num_max_components - 1, dtype=jnp.int32)
        final_loop_state, _ = (
            scanned_body(initial_state, indices_for_scan) @ "sbp_internal_scan_loop"
        )
        final_weights_array = final_loop_state.weights.at[num_max_components - 1].set(
            final_loop_state.remaining_stick
        )
        return StickBreakingResult(weights=final_weights_array)

    return sbp_instance_model


# --- CrossCat data strcture ---
@jax.tree_util.register_dataclass
@dataclass
class CrossCatHyperparams:
    alpha_view: float
    alpha_category: float
    bern_alpha: float
    bern_beta: float


@jax.tree_util.register_dataclass
@dataclass
class CrossCatLatents:
    view_weights: Float[Array, "max_views"]
    column_view_assignments: Integer[Array, "num_cols"]
    local_col_indices: Integer[Array, "num_cols"]
    num_cols_per_view: Integer[Array, "max_views"]
    category_weights_all_views: Float[Array, "max_views max_categories"]
    row_category_assignments_all_views: Integer[Array, "max_views num_rows"]
    bernoulli_params: Float[Array, "max_views max_categories max_cols_per_view"]


@jax.tree_util.register_dataclass
@dataclass
class CrossCatResult:
    data: Integer[Array, "num_rows num_cols"]
    latents: CrossCatLatents


# --- CrossCat生成モデル ---
def create_crosscat_model(
    num_rows_static: int,
    num_cols_static: int,
    max_views_static: int,
    max_categories_static: int,
    max_cols_per_view_static: int,
):
    sbp_for_views = create_sbp_model(max_views_static)
    sbp_for_categories = create_sbp_model(max_categories_static)

    @gen
    def crosscat_instance_model(hyperparams: CrossCatHyperparams) -> CrossCatResult:
        num_rows = num_rows_static
        num_cols = num_cols_static
        max_views = max_views_static
        max_categories = max_categories_static
        max_cols_per_view = max_cols_per_view_static

        view_sbr: StickBreakingResult = (
            sbp_for_views(hyperparams.alpha_view) @ "view_sbp_generation"
        )
        view_weights = view_sbr.weights

        _col_assignments_list = []
        for j_idx in range(num_cols):
            assignment = categorical(view_weights) @ f"column_view_assignment_{j_idx}"
            _col_assignments_list.append(assignment)
        column_view_assignments = jnp.stack(_col_assignments_list)
        column_view_assignments = jnp.reshape(column_view_assignments, (num_cols,))

        _local_col_indices = jnp.full(num_cols, -1, dtype=jnp.int32)
        _num_cols_per_view = jnp.bincount(
            column_view_assignments, length=max_views
        ).astype(jnp.int32)

        _current_local_idx_counters = jnp.zeros(max_views, dtype=jnp.int32)
        for j_global in range(num_cols):
            assigned_v = column_view_assignments[j_global]
            _local_col_indices = _local_col_indices.at[j_global].set(
                _current_local_idx_counters[assigned_v]
            )
            _current_local_idx_counters = _current_local_idx_counters.at[
                assigned_v
            ].add(1)

        _bernoulli_params = jnp.zeros((max_views, max_categories, max_cols_per_view))
        _row_category_assignments = jnp.zeros((max_views, num_rows), dtype=jnp.int32)
        _category_weights_all = jnp.zeros((max_views, max_categories))

        for v_idx in range(max_views):
            cat_sbr_loop: StickBreakingResult = (
                sbp_for_categories(hyperparams.alpha_category)
                @ f"category_sbp_view_{v_idx}"
            )
            cat_weights_v = cat_sbr_loop.weights
            _category_weights_all = _category_weights_all.at[v_idx, :].set(
                cat_weights_v
            )

            _row_assignments_for_view_v_list = []
            for r_idx in range(num_rows):
                assignment = (
                    categorical(cat_weights_v)
                    @ f"row_cat_assignment_view_{v_idx}_row_{r_idx}"
                )
                _row_assignments_for_view_v_list.append(assignment)

            _row_category_assignments = _row_category_assignments.at[v_idx, :].set(
                jnp.reshape(jnp.stack(_row_assignments_for_view_v_list), (num_rows,))
            )

            for c_idx in range(max_categories):
                for local_j_idx in range(max_cols_per_view):
                    p_val_beta = (
                        beta(hyperparams.bern_alpha, hyperparams.bern_beta)
                        @ f"p_v{v_idx}_c{c_idx}_lc{local_j_idx}"
                    )
                    _bernoulli_params = _bernoulli_params.at[
                        v_idx, c_idx, local_j_idx
                    ].set(p_val_beta)

        _data = jnp.zeros((num_rows, num_cols), dtype=jnp.int32)
        for r_idx in range(num_rows):
            for j_global_col in range(num_cols):
                assigned_view = column_view_assignments[j_global_col]
                assigned_category_for_row_in_view = _row_category_assignments[
                    assigned_view, r_idx
                ]
                local_col_idx_in_view = _local_col_indices[j_global_col]
                param_for_cell = _bernoulli_params[
                    assigned_view,
                    assigned_category_for_row_in_view,
                    local_col_idx_in_view,
                ]
                val = bernoulli(param_for_cell) @ f"data_r{r_idx}_c{j_global_col}"
                _data = _data.at[r_idx, j_global_col].set(val)

        latents = CrossCatLatents(
            view_weights=view_weights,
            column_view_assignments=column_view_assignments,
            local_col_indices=_local_col_indices,
            num_cols_per_view=_num_cols_per_view,
            category_weights_all_views=_category_weights_all,
            row_category_assignments_all_views=_row_category_assignments,
            bernoulli_params=_bernoulli_params,
        )
        return CrossCatResult(data=_data, latents=latents)

    return crosscat_instance_model


# --- data from ps2-problem3-soln.ipynb ---
test_dataset_raw = [
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [False, False, True, True, True, False],
    [False, False, True, True, True, False],
    [False, False, True, True, True, False],
    [False, False, True, True, True, False],
    [False, False, True, True, True, False],
]


# --- MCMC フレームワーク ---
def metropolis_hastings_move(mh_args, key):
    trace, model, proposal, proposal_args, observations = mh_args
    model_args = trace.get_args()
    argdiffs = Diff.no_change(model_args)
    proposal_args_forward = (trace, *proposal_args)

    key_propose, key_update, key_accept = jax.random.split(key, 3)
    fwd_choices, fwd_weight, _ = proposal.propose(key_propose, proposal_args_forward)

    new_trace, weight, _, discard = model.update(
        key_update, trace, fwd_choices, argdiffs
    )

    proposal_args_backward = (new_trace, *proposal_args)
    bwd_weight, _ = proposal.assess(discard, proposal_args_backward)

    alpha_mh = weight - fwd_weight + bwd_weight

    accepted_trace = jax.lax.cond(
        jnp.log(jax.random.uniform(key_accept)) < alpha_mh,
        lambda: new_trace,
        lambda: trace,
    )
    return (
        accepted_trace,
        model,
        proposal,
        proposal_args,
        observations,
    ), accepted_trace


def mh_sampler_loop(
    trace,
    model,
    proposal_fn,
    proposal_args_tuple,
    observations_choicemap,
    key,
    num_updates,
):
    mh_keys = jax.random.split(key, num_updates)
    initial_carry = (
        trace,
        model,
        proposal_fn,
        proposal_args_tuple,
        observations_choicemap,
    )

    last_carry, mh_chain_traces = jax.lax.scan(
        metropolis_hastings_move,
        initial_carry,
        mh_keys,
    )
    return last_carry[0], mh_chain_traces


# --- CrossCat proposal distribution ---
def create_propose_one_bernoulli_param(
    v_idx_static: int, c_idx_static: int, local_j_idx_static: int
):
    fixed_addr = f"p_v{v_idx_static}_c{c_idx_static}_lc{local_j_idx_static}"

    @gen
    def proposal_instance(
        current_trace: Trace, hyperparams_for_proposal: CrossCatHyperparams
    ):
        new_p_val = (
            beta(
                hyperparams_for_proposal.bern_alpha, hyperparams_for_proposal.bern_beta
            )
            @ fixed_addr
        )
        return new_p_val

    return proposal_instance


# --- MCMC ---
def crosscat_mh_kernel(
    trace,
    model,
    observations_choicemap,
    key,
    num_updates,
    proposal_fn_instance,  # ファクトリから生成された提案関数インスタンス
    hyperparams_for_prop: CrossCatHyperparams,
):
    proposal_args_tuple = (hyperparams_for_prop,)

    return mh_sampler_loop(
        trace,
        model,
        proposal_fn_instance,
        proposal_args_tuple,
        observations_choicemap,
        key,
        num_updates,
    )


def run_crosscat_inference(
    cc_model_instance,
    model_hyperparams: CrossCatHyperparams,
    observations: ChoiceMap,
    key: PRNGKeyArray,
    num_mcmc_steps: int,
    proposal_creator_fn,  # create_propose_one_bernoulli_param
    prop_v_idx: int,
    prop_c_idx: int,
    prop_lc_idx: int,
):
    key_init, key_mcmc = jax.random.split(key)

    specific_proposal_fn = proposal_creator_fn(prop_v_idx, prop_c_idx, prop_lc_idx)

    initial_trace, _ = cc_model_instance.importance(
        key_init, observations, (model_hyperparams,)
    )

    final_trace, trace_chain = crosscat_mh_kernel(
        initial_trace,
        cc_model_instance,
        observations,
        key_mcmc,
        num_mcmc_steps,
        specific_proposal_fn,
        model_hyperparams,
    )
    return final_trace, trace_chain


# --- Main ---
def test_crosscat_mcmc():
    observed_data_matrix = jnp.array(test_dataset_raw, dtype=jnp.int32)
    num_rows_val, num_cols_val = observed_data_matrix.shape

    print(
        f"MCMC Test: Using observed_data with shape: rows={num_rows_val}, cols={num_cols_val}"
    )
    print("Observed Data (first 5 rows):")
    print(observed_data_matrix[:5, :])
    print("-" * 30)

    obs_dict = {}
    for r in range(num_rows_val):
        for c_col in range(num_cols_val):
            obs_dict[f"data_r{r}_c{c_col}"] = observed_data_matrix[r, c_col]
    observations_choicemap = C.d(obs_dict)

    max_views_val = min(4, num_cols_val)
    max_categories_val = min(8, num_rows_val)
    max_cols_per_view_val = num_cols_val

    hyperparams_val = CrossCatHyperparams(
        alpha_view=1.0, alpha_category=1.0, bern_alpha=1.0, bern_beta=1.0
    )

    key = jax.random.PRNGKey(123)

    my_crosscat_model = create_crosscat_model(
        num_rows_static=num_rows_val,
        num_cols_static=num_cols_val,
        max_views_static=max_views_val,
        max_categories_static=max_categories_val,
        max_cols_per_view_static=max_cols_per_view_val,
    )

    num_mcmc_iterations = 100
    prop_v_idx_val = 0
    prop_c_idx_val = 0
    prop_lc_idx_val = 0

    key, subkey_inf = jax.random.split(key)
    final_trace, mcmc_chain = run_crosscat_inference(
        my_crosscat_model,
        hyperparams_val,
        observations_choicemap,
        subkey_inf,
        num_mcmc_iterations,
        create_propose_one_bernoulli_param,
        prop_v_idx_val,
        prop_c_idx_val,
        prop_lc_idx_val,
    )

    print("MCMC Inference Complete")
    print("=" * 30)
    if (
        final_trace is not None
        and hasattr(final_trace, "get_retval")
        and final_trace.get_retval() is not None
    ):
        final_latents: CrossCatLatents = final_trace.get_retval().latents
        print(f"  Trace Score: {final_trace.get_score()}")
        print(
            f"  View Weights (sum={jnp.sum(final_latents.view_weights):.2f}):\n{final_latents.view_weights}"
        )
        print(f"  Column View Assignments:\n{final_latents.column_view_assignments}")
        print(f"  Num Cols Per View:\n{final_latents.num_cols_per_view}")

        updated_param_addr = (
            f"p_v{prop_v_idx_val}_c{prop_c_idx_val}_lc{prop_lc_idx_val}"
        )
        final_choices = final_trace.get_choices()
        if final_choices is not None and updated_param_addr in final_choices:
            print(
                f"  Value of {updated_param_addr} after MCMC: {final_choices[updated_param_addr]}"
            )
        else:
            print(
                f"  Parameter {updated_param_addr} may not be active or was not in final trace choices (or final_choices is None)."
            )
    else:
        print("MCMC did not produce a valid final trace.")


if __name__ == "__main__":
    test_crosscat_mcmc()
