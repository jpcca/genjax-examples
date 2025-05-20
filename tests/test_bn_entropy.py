import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, vmap, nn, jit
import genjax
from genjax import gen, bernoulli, categorical, ChoiceMapBuilder as C
import functools
from functools import partial

# Import generic EEVI estimators
from src.eevi.estimator import (
    estimate_entropy_bounds_generic,
)  # ユーザーの環境に合わせてパスを調整してください

import pytest

# import numpy as np # NumPyは直接は使わないが、型ヒントや互換性のために残す

CSV_FILE_PATH = "tests/data/1-example/preprocessed.csv"
WEB_PAGE_START_COLUMN_INDEX = 3


FloatArray = jax.Array
PRNGKey = jax.random.PRNGKey
Arguments = tuple
Score = FloatArray
Weight = FloatArray


df_global = pd.read_csv(CSV_FILE_PATH)
web_page_columns_global = df_global.columns[WEB_PAGE_START_COLUMN_INDEX:].tolist()
num_web_pages_global = len(web_page_columns_global)


@gen
def sampling_sex_model(p_target_sex: float):
    is_target_sex = bernoulli(probs=p_target_sex) @ "sex_is_target"
    return is_target_sex


@gen
def sampling_age_model(age_category_probs: jnp.ndarray):
    age_idx = categorical(probs=age_category_probs) @ "age_idx"
    return age_idx


@gen
def bn_web_pages_model(sex_idx: int, age_idx: int, web_pages_cpt: jnp.ndarray):
    prob_watch_vector = jnp.clip(web_pages_cpt[:, sex_idx, age_idx], 1e-7, 1.0 - 1e-7)
    watched_states = []
    for i_loop_var in range(web_pages_cpt.shape[0]):  # num_web_pages_global と同じはず
        page_watched = (
            bernoulli(probs=prob_watch_vector[i_loop_var]) @ f"web_page_{i_loop_var}"
        )
        watched_states.append(page_watched)
    return jnp.stack(watched_states)


# --- preprocessing data and calculating CPT ---
def preprocess_data_and_calculate_params(df_input: pd.DataFrame):
    local_df = df_input.copy()
    valid_web_page_columns = [
        col for col in web_page_columns_global if col in local_df.columns
    ]

    for page_col in valid_web_page_columns:
        if local_df[page_col].dtype == bool:
            local_df[f"{page_col}_numeric"] = local_df[page_col].astype(int)
        elif (
            pd.api.types.is_numeric_dtype(local_df[page_col])
            and local_df[page_col].isin([0, 1]).all()
        ):
            local_df[f"{page_col}_numeric"] = local_df[page_col]
        elif local_df[page_col].astype(str).str.lower().isin(["true", "false"]).all():
            local_df[f"{page_col}_numeric"] = (
                local_df[page_col].astype(str).str.lower().map({"true": 1, "false": 0})
            )
        else:
            local_df[f"{page_col}_numeric"] = 0

    sex_categories = sorted(local_df["sex"].unique())
    sex_map = {category: i for i, category in enumerate(sex_categories)}
    local_df["sex_idx"] = local_df["sex"].map(sex_map)
    num_sex_categories = len(sex_categories)

    age_categories = sorted(local_df["age"].unique())
    age_map = {category: i for i, category in enumerate(age_categories)}
    local_df["age_idx"] = local_df["age"].map(age_map)
    num_age_categories = len(age_categories)

    web_pages_cpt = jnp.zeros(
        (num_web_pages_global, num_sex_categories, num_age_categories)
    )
    for page_idx, page_col_global_name in enumerate(web_page_columns_global):
        numeric_col_name = f"{page_col_global_name}_numeric"
        if numeric_col_name not in local_df.columns:
            continue
        cpt_series = local_df.groupby(["sex_idx", "age_idx"], observed=False)[
            numeric_col_name
        ].mean()
        for s_idx in range(num_sex_categories):
            for a_idx in range(num_age_categories):
                value = cpt_series.get((s_idx, a_idx), 0.0)
                web_pages_cpt = web_pages_cpt.at[page_idx, s_idx, a_idx].set(value)

    sex_counts = local_df["sex_idx"].value_counts(dropna=False).sort_index()
    p_sex_vector = jnp.array(
        [sex_counts.get(i, 0) / len(local_df) for i in range(num_sex_categories)]
    )

    age_counts = local_df["age_idx"].value_counts(dropna=False).sort_index()
    p_age_vector = jnp.array(
        [age_counts.get(i, 0) / len(local_df) for i in range(num_age_categories)]
    )

    return (
        sex_map,
        age_map,
        web_pages_cpt,
        p_sex_vector,
        p_age_vector,
        num_sex_categories,
        num_age_categories,
    )


# --- Analytical MI ---
def get_analytical_probabilities(
    p_sex_vector: jnp.ndarray,
    p_age_vector: jnp.ndarray,
    web_pages_cpt: jnp.ndarray,
    target_page_idx: int,
    num_sex_categories: int,
    num_age_categories: int,
):
    p_wk_one = 0.0
    for s_idx in range(num_sex_categories):
        for a_idx in range(num_age_categories):
            p_wk_one += (
                web_pages_cpt[target_page_idx, s_idx, a_idx]
                * p_sex_vector[s_idx]
                * p_age_vector[a_idx]
            )
    p_wk_dist = jnp.array([1.0 - p_wk_one, p_wk_one])

    p_s_wk_joint = jnp.zeros((num_sex_categories, 2))  # Assuming Wk is binary
    for s_idx in range(num_sex_categories):
        p_s_wk_one_given_s = 0.0
        for a_idx in range(num_age_categories):
            p_s_wk_one_given_s += (
                web_pages_cpt[target_page_idx, s_idx, a_idx] * p_age_vector[a_idx]
            )
        p_s_wk_joint = p_s_wk_joint.at[s_idx, 1].set(
            p_s_wk_one_given_s * p_sex_vector[s_idx]
        )
        p_s_wk_joint = p_s_wk_joint.at[s_idx, 0].set(
            (1.0 - p_s_wk_one_given_s) * p_sex_vector[s_idx]
        )
    return p_sex_vector, p_wk_dist, p_s_wk_joint


@jit
def calculate_entropy_discrete(prob_dist: jnp.ndarray) -> FloatArray:
    prob_dist_clipped = jnp.clip(prob_dist, 1e-9, 1.0)  # Avoid log(0)
    return -jnp.sum(prob_dist_clipped * jnp.log2(prob_dist_clipped))


def calculate_analytical_mi(
    p_sex_vector: jnp.ndarray,
    p_age_vector: jnp.ndarray,
    web_pages_cpt: jnp.ndarray,
    target_page_idx: int,
    num_sex_categories: int,
    num_age_categories: int,
) -> FloatArray:
    p_s, p_wk, p_s_wk = get_analytical_probabilities(
        p_sex_vector,
        p_age_vector,
        web_pages_cpt,
        target_page_idx,
        num_sex_categories,
        num_age_categories,
    )
    H_S = calculate_entropy_discrete(p_s)
    H_Wk = calculate_entropy_discrete(p_wk)
    H_S_Wk = calculate_entropy_discrete(p_s_wk.flatten())
    mi = H_S + H_Wk - H_S_Wk
    return mi, H_S, H_Wk, H_S_Wk


# --- EEVI ---
@gen
def full_bn_model(
    p_sex_vector: jnp.ndarray, p_age_vector: jnp.ndarray, web_pages_cpt: jnp.ndarray
):
    sex_idx = categorical(logits=jnp.log(jnp.clip(p_sex_vector, 1e-9))) @ "sex"
    age_idx = categorical(logits=jnp.log(jnp.clip(p_age_vector, 1e-9))) @ "age"

    prob_watch_vector = jnp.clip(web_pages_cpt[:, sex_idx, age_idx], 1e-7, 1.0 - 1e-7)

    watched_states_list = []
    for i in range(num_web_pages_global):
        page_watched = bernoulli(probs=prob_watch_vector[i]) @ f"web_page_{i}"
        watched_states_list.append(page_watched)
    return sex_idx, age_idx, jnp.stack(watched_states_list)


def make_base_proposal_H_S(static_num_web_pages: int):
    @gen
    def base_proposal_H_S_specialized(
        sex_idx: int,
        p_age_marginal: jnp.ndarray,
        web_pages_cpt: jnp.ndarray,
    ):
        age_idx = (
            categorical(logits=jnp.log(jnp.clip(p_age_marginal, 1e-9))) @ "age_q0_S"
        )
        prob_watch_vector = jnp.clip(
            web_pages_cpt[:static_num_web_pages, sex_idx, age_idx],
            1e-7,
            1.0 - 1e-7,
        )
        watched_states_list = []
        for i in range(static_num_web_pages):
            page_watched = bernoulli(probs=prob_watch_vector[i]) @ f"web_page_q0_S_{i}"
            watched_states_list.append(page_watched)
        return age_idx, jnp.stack(watched_states_list)

    return base_proposal_H_S_specialized


def make_base_proposal_H_Wk(
    static_target_page_idx: int,
    static_num_web_pages: int,
    num_sex_categories: int,
    num_age_categories: int,
):
    @gen
    def base_proposal_H_Wk_specialized(
        wk_idx: int,
        p_sex_marginal: jnp.ndarray,
        p_age_marginal: jnp.ndarray,
        web_pages_cpt: jnp.ndarray,
    ):
        log_probs_sex_given_wk = jnp.zeros(num_sex_categories)
        for s_loop_idx in range(num_sex_categories):
            prob_wk_given_s = 0.0
            for a_loop_idx in range(num_age_categories):
                prob_wk_given_s_a_raw = web_pages_cpt[
                    static_target_page_idx, s_loop_idx, a_loop_idx
                ]
                # JAX-friendly conditional assignment
                prob_wk_given_s_a = jnp.where(
                    jnp.equal(wk_idx, 0),
                    1.0 - prob_wk_given_s_a_raw,
                    prob_wk_given_s_a_raw,
                )
                prob_wk_given_s += prob_wk_given_s_a * p_age_marginal[a_loop_idx]
            log_probs_sex_given_wk = log_probs_sex_given_wk.at[s_loop_idx].set(
                jnp.log(jnp.clip(prob_wk_given_s * p_sex_marginal[s_loop_idx], 1e-9))
            )
        sex_idx = categorical(logits=log_probs_sex_given_wk) @ "sex_q0_Wk"

        log_probs_age_given_s_wk = jnp.zeros(num_age_categories)
        for a_loop_idx in range(num_age_categories):
            prob_wk_given_s_a_raw = web_pages_cpt[
                static_target_page_idx, sex_idx, a_loop_idx
            ]
            prob_wk_given_s_a = jnp.where(
                jnp.equal(wk_idx, 0), 1.0 - prob_wk_given_s_a_raw, prob_wk_given_s_a_raw
            )
            log_probs_age_given_s_wk = log_probs_age_given_s_wk.at[a_loop_idx].set(
                jnp.log(jnp.clip(prob_wk_given_s_a * p_age_marginal[a_loop_idx], 1e-9))
            )
        age_idx = categorical(logits=log_probs_age_given_s_wk) @ "age_q0_Wk"

        other_webpages_states_list = []
        prob_watch_vector_all = jnp.clip(
            web_pages_cpt[:static_num_web_pages, sex_idx, age_idx], 1e-7, 1.0 - 1e-7
        )
        for i in range(static_num_web_pages):
            if i == static_target_page_idx:
                continue
            page_watched = (
                bernoulli(probs=prob_watch_vector_all[i]) @ f"web_page_q0_Wk_{i}"
            )
            other_webpages_states_list.append(page_watched)

        return (
            sex_idx,
            age_idx,
            (
                jnp.stack(other_webpages_states_list)
                if other_webpages_states_list
                else jnp.array([], dtype=jnp.int32)
            ),  # Ensure dtype for empty
        )

    return base_proposal_H_Wk_specialized


def make_base_proposal_H_SWk(
    static_target_page_idx: int, static_num_web_pages: int, num_age_categories: int
):
    @gen
    def base_proposal_H_SWk_specialized(
        s_wk_tuple: tuple[FloatArray, FloatArray],
        p_age_marginal: jnp.ndarray,
        web_pages_cpt: jnp.ndarray,
    ):
        sex_idx = s_wk_tuple[0].astype(jnp.int32)
        wk_idx = s_wk_tuple[1].astype(jnp.int32)

        log_probs_age_given_s_wk = jnp.zeros(num_age_categories)
        for a_loop_idx in range(num_age_categories):
            prob_wk_given_s_a_raw = web_pages_cpt[
                static_target_page_idx, sex_idx, a_loop_idx
            ]
            prob_wk_given_s_a = jnp.where(
                jnp.equal(wk_idx, 0), 1.0 - prob_wk_given_s_a_raw, prob_wk_given_s_a_raw
            )
            log_probs_age_given_s_wk = log_probs_age_given_s_wk.at[a_loop_idx].set(
                jnp.log(jnp.clip(prob_wk_given_s_a * p_age_marginal[a_loop_idx], 1e-9))
            )
        age_idx = categorical(logits=log_probs_age_given_s_wk) @ "age_q0_SWk"

        other_webpages_states_list = []
        prob_watch_vector_all = jnp.clip(
            web_pages_cpt[:static_num_web_pages, sex_idx, age_idx], 1e-7, 1.0 - 1e-7
        )
        for i in range(static_num_web_pages):
            if i == static_target_page_idx:
                continue
            page_watched = (
                bernoulli(probs=prob_watch_vector_all[i]) @ f"web_page_q0_SWk_{i}"
            )
            other_webpages_states_list.append(page_watched)

        return age_idx, (
            jnp.stack(other_webpages_states_list)
            if other_webpages_states_list
            else jnp.array([], dtype=jnp.int32)  # Ensure dtype for empty
        )

    return base_proposal_H_SWk_specialized


# --- Helper functions for generic EEVI estimator --- (No changes below this line from last version)
@jit
def reconstruct_z_H_S_values(
    x_sample: tuple[FloatArray, FloatArray],
    y_sample_sex: FloatArray,
) -> tuple:
    age_idx_q0, webpages_q0 = x_sample
    return (y_sample_sex.astype(jnp.int32), age_idx_q0.astype(jnp.int32), webpages_q0)


@partial(jit, static_argnums=(2,))
def reconstruct_z_H_Wk_values(
    x_sample: tuple[FloatArray, FloatArray, FloatArray],
    y_sample_wk: FloatArray,
    target_page_idx: int,
) -> tuple:
    sex_idx_q0, age_idx_q0, other_pages_q0 = x_sample
    all_webpages = jnp.zeros(num_web_pages_global, dtype=jnp.int32)
    current_other_idx = 0
    for i in range(num_web_pages_global):
        if i == target_page_idx:
            all_webpages = all_webpages.at[i].set(y_sample_wk.astype(jnp.int32))
        else:
            if other_pages_q0.size > 0 and current_other_idx < other_pages_q0.shape[0]:
                all_webpages = all_webpages.at[i].set(
                    other_pages_q0[current_other_idx].astype(jnp.int32)
                )
                current_other_idx += 1
    return (sex_idx_q0.astype(jnp.int32), age_idx_q0.astype(jnp.int32), all_webpages)


@partial(jit, static_argnums=(2,))
def reconstruct_z_H_SWk_values(
    x_sample: tuple[FloatArray, FloatArray],
    y_sample_s_wk: tuple[FloatArray, FloatArray],
    target_page_idx: int,
) -> tuple:
    age_idx_q0, other_pages_q0 = x_sample
    sex_val_y, wk_val_y = y_sample_s_wk[0].astype(jnp.int32), y_sample_s_wk[1].astype(
        jnp.int32
    )
    all_webpages = jnp.zeros(num_web_pages_global, dtype=jnp.int32)
    current_other_idx = 0
    for i in range(num_web_pages_global):
        if i == target_page_idx:
            all_webpages = all_webpages.at[i].set(wk_val_y)
        else:
            if other_pages_q0.size > 0 and current_other_idx < other_pages_q0.shape[0]:
                all_webpages = all_webpages.at[i].set(
                    other_pages_q0[current_other_idx].astype(jnp.int32)
                )
                current_other_idx += 1
    return (sex_val_y, age_idx_q0.astype(jnp.int32), all_webpages)


def build_p_choices_for_bn(
    z_values: tuple, num_web_pages_for_p_assess: int
) -> genjax.ChoiceMap:
    sex_val, age_val, webpages_val = z_values
    choices = C.n()
    choices = choices | C["sex"].set(sex_val.astype(jnp.int32))
    choices = choices | C["age"].set(age_val.astype(jnp.int32))
    max_pages_to_set = min(webpages_val.shape[0], num_web_pages_for_p_assess)
    for i_wp in range(max_pages_to_set):
        choices = choices | C[f"web_page_{i_wp}"].set(
            webpages_val[i_wp].astype(jnp.int32)
        )
    return choices


def build_q0_H_S_choices(x_prime_values: tuple) -> genjax.ChoiceMap:
    age_val_xp, webpages_val_xp = x_prime_values
    choices = C.n()
    choices = choices | C["age_q0_S"].set(age_val_xp.astype(jnp.int32))
    for i, wp_val in enumerate(webpages_val_xp):
        choices = choices | C[f"web_page_q0_S_{i}"].set(wp_val.astype(jnp.int32))
    return choices


def build_q0_H_Wk_choices(
    x_prime_values: tuple, target_page_idx: int, num_web_pages: int
) -> genjax.ChoiceMap:
    sex_val_xp, age_val_xp, other_wps_xp = x_prime_values
    choices = C.n()
    choices = choices | C["sex_q0_Wk"].set(sex_val_xp.astype(jnp.int32))
    choices = choices | C["age_q0_Wk"].set(age_val_xp.astype(jnp.int32))
    other_page_counter = 0
    for i in range(num_web_pages):
        if i == target_page_idx:
            continue
        if other_wps_xp.size > 0 and other_page_counter < other_wps_xp.shape[0]:
            choices = choices | C[f"web_page_q0_Wk_{i}"].set(
                other_wps_xp[other_page_counter].astype(jnp.int32)
            )
            other_page_counter += 1
    return choices


def build_q0_H_SWk_choices(
    x_prime_values: tuple, target_page_idx: int, num_web_pages: int
) -> genjax.ChoiceMap:
    age_val_xp, other_wps_xp = x_prime_values
    choices = C.n()
    choices = choices | C["age_q0_SWk"].set(age_val_xp.astype(jnp.int32))
    other_page_counter = 0
    for i in range(num_web_pages):
        if i == target_page_idx:
            continue
        if other_wps_xp.size > 0 and other_page_counter < other_wps_xp.shape[0]:
            choices = choices | C[f"web_page_q0_SWk_{i}"].set(
                other_wps_xp[other_page_counter].astype(jnp.int32)
            )
            other_page_counter += 1
    return choices


def sample_y_for_bn_entropy(
    key: PRNGKey,
    n_samples: int,
    y_var_type: str,
    p_sex_true_dist: jnp.ndarray,
    p_age_true_dist: jnp.ndarray,
    web_pages_cpt_true: jnp.ndarray,
    target_page_idx_for_y: int,
    num_sex_cat_true: int,
    num_age_cat_true: int,
) -> tuple[FloatArray, ...] | FloatArray:
    if y_var_type == "S":
        return random.categorical(
            key, logits=jnp.log(jnp.clip(p_sex_true_dist, 1e-9)), shape=(n_samples,)
        )
    elif y_var_type == "Wk":
        _, p_wk_dist, _ = get_analytical_probabilities(
            p_sex_true_dist,
            p_age_true_dist,
            web_pages_cpt_true,
            target_page_idx_for_y,
            num_sex_cat_true,
            num_age_cat_true,
        )
        return random.categorical(
            key, logits=jnp.log(jnp.clip(p_wk_dist, 1e-9)), shape=(n_samples,)
        )
    elif y_var_type == "SWk":
        _, _, p_s_wk_joint = get_analytical_probabilities(
            p_sex_true_dist,
            p_age_true_dist,
            web_pages_cpt_true,
            target_page_idx_for_y,
            num_sex_cat_true,
            num_age_cat_true,
        )
        p_s_wk_flat = p_s_wk_joint.flatten()
        num_wk_categories = p_s_wk_joint.shape[1]
        sampled_indices_flat = random.categorical(
            key, logits=jnp.log(jnp.clip(p_s_wk_flat, 1e-9)), shape=(n_samples,)
        )
        s_indices = sampled_indices_flat // num_wk_categories
        wk_states = sampled_indices_flat % num_wk_categories
        return (s_indices.astype(jnp.float32), wk_states.astype(jnp.float32))
    raise ValueError(f"Unknown y_var_type: {y_var_type}")


def curried_extract_y_prime_H_S(joint_choice: genjax.ChoiceMap):
    return joint_choice["sex"]


def curried_extract_x_prime_H_S(joint_choice: genjax.ChoiceMap, num_web_pages: int):
    age_val = joint_choice["age"]
    webpages_val_list = []
    for i in range(num_web_pages):
        if f"web_page_{i}" in joint_choice:  # Check if key exists
            webpages_val_list.append(joint_choice[f"web_page_{i}"])
        else:
            webpages_val_list.append(0)  # Default if not found
    return (age_val, jnp.array(webpages_val_list, dtype=jnp.int32))


def curried_extract_y_prime_H_Wk(joint_choice: genjax.ChoiceMap, target_page_idx: int):
    return joint_choice[f"web_page_{target_page_idx}"]


def curried_extract_x_prime_H_Wk(
    joint_choice: genjax.ChoiceMap, target_page_idx: int, num_web_pages: int
):
    sex_val = joint_choice["sex"]
    age_val = joint_choice["age"]
    other_pages_list = []
    for i in range(num_web_pages):
        if i == target_page_idx:
            continue
        if f"web_page_{i}" in joint_choice:
            other_pages_list.append(joint_choice[f"web_page_{i}"])
        else:
            other_pages_list.append(0)
    return (
        sex_val,
        age_val,
        (
            jnp.array(other_pages_list, dtype=jnp.int32)
            if other_pages_list
            else jnp.array([], dtype=jnp.int32)
        ),
    )


def curried_extract_y_prime_H_SWk(joint_choice: genjax.ChoiceMap, target_page_idx: int):
    sex_val = joint_choice["sex"]
    wk_val = joint_choice[f"web_page_{target_page_idx}"]
    return (sex_val.astype(jnp.float32), wk_val.astype(jnp.float32))


def curried_extract_x_prime_H_SWk(
    joint_choice: genjax.ChoiceMap, target_page_idx: int, num_web_pages: int
):
    age_val = joint_choice["age"]
    other_pages_list = []
    for i in range(num_web_pages):
        if i == target_page_idx:
            continue
        if f"web_page_{i}" in joint_choice:
            other_pages_list.append(joint_choice[f"web_page_{i}"])
        else:
            other_pages_list.append(0)
    return (
        age_val,
        (
            jnp.array(other_pages_list, dtype=jnp.int32)
            if other_pages_list
            else jnp.array([], dtype=jnp.int32)
        ),
    )


def test_mutual_information_estimation_with_e_evi():
    (
        sex_map_gl,
        age_map_gl,
        web_cpt_gl,
        p_sex_gl,
        p_age_gl,
        n_sex_gl,
        n_age_gl,
    ) = preprocess_data_and_calculate_params(df_global)

    p_sex_gl = jnp.clip(p_sex_gl, 1e-9, 1.0)
    p_age_gl = jnp.clip(p_age_gl, 1e-9, 1.0)
    web_cpt_gl = jnp.clip(web_cpt_gl, 1e-7, 1.0 - 1e-7)

    print(f"P(Sex) from data = {p_sex_gl}")
    print(f"P(Age) from data = {p_age_gl}")

    TARGET_PAGE_FOR_MI = 0

    print("\n--- Analytical MI ---")
    analytical_mi_val, an_H_S, an_H_Wk, an_H_SWk = calculate_analytical_mi(
        p_sex_gl, p_age_gl, web_cpt_gl, TARGET_PAGE_FOR_MI, n_sex_gl, n_age_gl
    )
    print(f"  Analytical H(S)      = {an_H_S:.4f} bits")
    print(f"  Analytical H(W{TARGET_PAGE_FOR_MI})    = {an_H_Wk:.4f} bits")
    print(f"  Analytical H(S,W{TARGET_PAGE_FOR_MI}) = {an_H_SWk:.4f} bits")
    print(f"  Analytical MI(S;W{TARGET_PAGE_FOR_MI}) = {analytical_mi_val:.4f} bits")

    print("\n--- EEVI MI ---")
    N_OUTER = 500
    P_SIR = 50
    MAIN_KEY = random.PRNGKey(12345)
    p_model_eevi_final = full_bn_model

    p_model_args_eevi = (p_sex_gl, p_age_gl, web_cpt_gl)
    build_p_choices_fn_curried = functools.partial(
        build_p_choices_for_bn, num_web_pages_for_p_assess=num_web_pages_global
    )

    # --- Estimate H(S) ---
    key_hs, MAIN_KEY = random.split(MAIN_KEY)
    print(f"Estimating H(S) with EEVI (N_outer={N_OUTER}, P_sir={P_SIR})...")
    q0_model_H_S_spec = make_base_proposal_H_S(num_web_pages_global)
    q0_args_H_S = (p_age_gl, web_cpt_gl)
    reconstruct_z_fn_H_S = reconstruct_z_H_S_values
    extract_y_prime_fn_H_S = curried_extract_y_prime_H_S
    extract_x_prime_values_fn_H_S = functools.partial(
        curried_extract_x_prime_H_S, num_web_pages=num_web_pages_global
    )
    build_q0_choices_fn_H_S = build_q0_H_S_choices
    y_sampling_args_H_S = ("S", p_sex_gl, p_age_gl, web_cpt_gl, -1, n_sex_gl, n_age_gl)
    analytical_H_Y_args_H_S = (p_sex_gl,)
    an_hs_val, tilde_H_S, hat_H_S = estimate_entropy_bounds_generic(
        key_hs,
        N_OUTER,
        P_SIR,
        p_model_eevi_final,
        p_model_args_eevi,
        q0_model_H_S_spec,
        q0_args_H_S,
        reconstruct_z_fn_H_S,
        build_p_choices_fn_curried,
        extract_y_prime_fn_H_S,
        extract_x_prime_values_fn_H_S,
        build_q0_choices_fn_H_S,
        sample_y_for_bn_entropy,
        y_sampling_args_H_S,
        analytical_H_Y_fn=calculate_entropy_discrete,
        analytical_H_Y_args=analytical_H_Y_args_H_S,
    )
    print(
        f"  EEVI H(S): Analytical={an_hs_val:.4f}, Lower={tilde_H_S:.4f}, Upper={hat_H_S:.4f}"
    )

    # --- Estimate H(Wk) ---
    key_hwk, MAIN_KEY = random.split(MAIN_KEY)
    print(
        f"Estimating H(W{TARGET_PAGE_FOR_MI}) with EEVI (N_outer={N_OUTER}, P_sir={P_SIR})..."
    )
    q0_model_H_Wk_spec = make_base_proposal_H_Wk(
        TARGET_PAGE_FOR_MI, num_web_pages_global, n_sex_gl, n_age_gl
    )
    q0_args_H_Wk = (p_sex_gl, p_age_gl, web_cpt_gl)
    reconstruct_z_fn_H_Wk = functools.partial(
        reconstruct_z_H_Wk_values, target_page_idx=TARGET_PAGE_FOR_MI
    )
    extract_y_prime_fn_H_Wk = functools.partial(
        curried_extract_y_prime_H_Wk, target_page_idx=TARGET_PAGE_FOR_MI
    )
    extract_x_prime_values_fn_H_Wk = functools.partial(
        curried_extract_x_prime_H_Wk,
        target_page_idx=TARGET_PAGE_FOR_MI,
        num_web_pages=num_web_pages_global,
    )
    build_q0_choices_fn_H_Wk = functools.partial(
        build_q0_H_Wk_choices,
        target_page_idx=TARGET_PAGE_FOR_MI,
        num_web_pages=num_web_pages_global,
    )
    y_sampling_args_H_Wk = (
        "Wk",
        p_sex_gl,
        p_age_gl,
        web_cpt_gl,
        TARGET_PAGE_FOR_MI,
        n_sex_gl,
        n_age_gl,
    )
    _, p_wk_dist_an, _ = get_analytical_probabilities(
        p_sex_gl, p_age_gl, web_cpt_gl, TARGET_PAGE_FOR_MI, n_sex_gl, n_age_gl
    )
    analytical_H_Y_args_H_Wk = (p_wk_dist_an,)
    an_hwk_val, tilde_H_Wk, hat_H_Wk = estimate_entropy_bounds_generic(
        key_hwk,
        N_OUTER,
        P_SIR,
        p_model_eevi_final,
        p_model_args_eevi,
        q0_model_H_Wk_spec,
        q0_args_H_Wk,
        reconstruct_z_fn_H_Wk,
        build_p_choices_fn_curried,
        extract_y_prime_fn_H_Wk,
        extract_x_prime_values_fn_H_Wk,
        build_q0_choices_fn_H_Wk,
        sample_y_for_bn_entropy,
        y_sampling_args_H_Wk,
        analytical_H_Y_fn=calculate_entropy_discrete,
        analytical_H_Y_args=analytical_H_Y_args_H_Wk,
    )
    print(
        f"  EEVI H(W{TARGET_PAGE_FOR_MI}): Analytical={an_hwk_val:.4f}, Lower={tilde_H_Wk:.4f}, Upper={hat_H_Wk:.4f}"
    )

    # --- Estimate H(S,Wk) ---
    key_hswk, MAIN_KEY = random.split(MAIN_KEY)
    print(
        f"Estimating H(S,W{TARGET_PAGE_FOR_MI}) with EEVI (N_outer={N_OUTER}, P_sir={P_SIR})..."
    )
    q0_model_H_SWk_spec = make_base_proposal_H_SWk(
        TARGET_PAGE_FOR_MI, num_web_pages_global, n_age_gl
    )
    q0_args_H_SWk = (p_age_gl, web_cpt_gl)
    reconstruct_z_fn_H_SWk = functools.partial(
        reconstruct_z_H_SWk_values, target_page_idx=TARGET_PAGE_FOR_MI
    )
    extract_y_prime_fn_H_SWk = functools.partial(
        curried_extract_y_prime_H_SWk, target_page_idx=TARGET_PAGE_FOR_MI
    )
    extract_x_prime_values_fn_H_SWk = functools.partial(
        curried_extract_x_prime_H_SWk,
        target_page_idx=TARGET_PAGE_FOR_MI,
        num_web_pages=num_web_pages_global,
    )
    build_q0_choices_fn_H_SWk = functools.partial(
        build_q0_H_SWk_choices,
        target_page_idx=TARGET_PAGE_FOR_MI,
        num_web_pages=num_web_pages_global,
    )
    y_sampling_args_H_SWk = (
        "SWk",
        p_sex_gl,
        p_age_gl,
        web_cpt_gl,
        TARGET_PAGE_FOR_MI,
        n_sex_gl,
        n_age_gl,
    )
    _, _, p_s_wk_joint_an = get_analytical_probabilities(
        p_sex_gl, p_age_gl, web_cpt_gl, TARGET_PAGE_FOR_MI, n_sex_gl, n_age_gl
    )
    analytical_H_Y_args_H_SWk = (p_s_wk_joint_an.flatten(),)
    an_hswk_val, tilde_H_SWk, hat_H_SWk = estimate_entropy_bounds_generic(
        key_hswk,
        N_OUTER,
        P_SIR,
        p_model_eevi_final,
        p_model_args_eevi,
        q0_model_H_SWk_spec,
        q0_args_H_SWk,
        reconstruct_z_fn_H_SWk,
        build_p_choices_fn_curried,
        extract_y_prime_fn_H_SWk,
        extract_x_prime_values_fn_H_SWk,
        build_q0_choices_fn_H_SWk,
        sample_y_for_bn_entropy,
        y_sampling_args_H_SWk,
        analytical_H_Y_fn=calculate_entropy_discrete,
        analytical_H_Y_args=analytical_H_Y_args_H_SWk,
    )
    print(
        f"  EEVI H(S,W{TARGET_PAGE_FOR_MI}): Analytical={an_hswk_val:.4f}, Lower={tilde_H_SWk:.4f}, Upper={hat_H_SWk:.4f}"
    )

    tilde_MI_eevi = tilde_H_S + tilde_H_Wk - hat_H_SWk
    hat_MI_eevi = hat_H_S + hat_H_Wk - tilde_H_SWk
    print(
        f"\n  EEVI MI(S;W{TARGET_PAGE_FOR_MI}) Interval: [{tilde_MI_eevi:.4f}, {hat_MI_eevi:.4f}] bits"
    )
    print(
        f"  Analytical MI(S;W{TARGET_PAGE_FOR_MI})      : {analytical_mi_val:.4f} bits"
    )

    tolerance = 0.05
    assert (
        (tilde_MI_eevi - tolerance) <= analytical_mi_val <= (hat_MI_eevi + tolerance)
    ), (
        f"Analytical MI {analytical_mi_val:.4f} not in EEVI interval [{tilde_MI_eevi:.4f}, {hat_MI_eevi:.4f}] with tolerance {tolerance}. "
        f"Lower diff: {analytical_mi_val - tilde_MI_eevi:.4f}, Upper diff: {hat_MI_eevi - analytical_mi_val:.4f}"
    )
