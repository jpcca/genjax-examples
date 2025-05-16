import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, vmap, nn, jit
import genjax
from genjax import gen, bernoulli, categorical, ChoiceMapBuilder as C
import functools
from functools import partial

# Import generic EEVI estimators
from src.eevi.estimator import estimate_entropy_bounds_generic

import pytest
import numpy as np

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
    for i_loop_var in range(web_pages_cpt.shape[0]):
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

    sex_counts = local_df["sex_idx"].value_counts().sort_index()
    p_sex_vector = jnp.array(
        [sex_counts.get(i, 0) / len(local_df) for i in range(num_sex_categories)]
    )

    age_counts = local_df["age_idx"].value_counts().sort_index()
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

    p_s_wk_joint = jnp.zeros((num_sex_categories, 2))
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
    prob_dist_clipped = jnp.clip(prob_dist, 1e-9, 1.0)
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
# Full BN model p(Sex, Age, WebPage_0, ..., WebPage_N-1)
@gen
def full_bn_model(
    p_sex_vector: jnp.ndarray, p_age_vector: jnp.ndarray, web_pages_cpt: jnp.ndarray
):
    sex_idx = categorical(probs=p_sex_vector) @ "sex"
    age_idx = categorical(probs=p_age_vector) @ "age"
    watched_states_list = []
    # Ensure web_pages_cpt access is within bounds if its first dim is num_web_pages_global
    prob_watch_vector = jnp.clip(web_pages_cpt[:, sex_idx, age_idx], 1e-7, 1.0 - 1e-7)
    for i in range(
        web_pages_cpt.shape[0]
    ):  # Assumes web_pages_cpt.shape[0] is num_web_pages_global
        page_watched = bernoulli(probs=prob_watch_vector[i]) @ f"web_page_{i}"
        watched_states_list.append(page_watched)
    return sex_idx, age_idx, jnp.stack(watched_states_list)


# Base proposal for H(S): q0(Age, WebPages | Sex=s_fixed)
def make_base_proposal_H_S(static_fixed_sex_idx: int, static_num_web_pages: int):
    @gen
    def base_proposal_H_S_specialized(
        p_age_vector: jnp.ndarray,  # p(Age)
        web_pages_cpt: jnp.ndarray,  # CPT P(WebPage_i | Sex, Age)
    ):
        age_idx = categorical(probs=p_age_vector) @ "age_q0_S"
        watched_states_list = []
        prob_watch_vector = jnp.clip(
            web_pages_cpt[:static_num_web_pages, static_fixed_sex_idx, age_idx],
            1e-7,
            1.0 - 1e-7,
        )
        for i in range(static_num_web_pages):
            page_watched = bernoulli(probs=prob_watch_vector[i]) @ f"web_page_q0_S_{i}"
            watched_states_list.append(page_watched)
        return age_idx, jnp.stack(
            watched_states_list
        )  # Returns (age_idx, web_pages_array)

    return base_proposal_H_S_specialized


# Base proposal for H(Wk): q0(Sex, Age, WebPages_{\Wk} | Wk=wk_fixed)
# Note: Wk is fixed by conditioning in p(z|Wk), so q0 samples other variables
def make_base_proposal_H_Wk(static_target_page_idx: int, static_num_web_pages: int):
    @gen
    def base_proposal_H_Wk_specialized(
        p_sex_vector: jnp.ndarray,  # p(Sex)
        p_age_vector: jnp.ndarray,  # p(Age)
        web_pages_cpt: jnp.ndarray,  # CPT P(WebPage_i | Sex, Age)
    ):
        sex_idx = categorical(probs=p_sex_vector) @ "sex_q0_Wk"
        age_idx = categorical(probs=p_age_vector) @ "age_q0_Wk"
        other_webpages_states_list = []
        # Probabilities for ALL web pages given sampled sex_idx, age_idx
        prob_watch_vector_all = jnp.clip(
            web_pages_cpt[:static_num_web_pages, sex_idx, age_idx], 1e-7, 1.0 - 1e-7
        )
        for i in range(static_num_web_pages):
            if i == static_target_page_idx:  # Wk is fixed, so don't sample it in q0
                continue
            # Sample other web pages
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
                else jnp.array([])  # Handle case if num_web_pages is 1
            ),
        )  # Returns (sex_idx, age_idx, other_webpages_array)

    return base_proposal_H_Wk_specialized


# Base proposal for H(S,Wk): q0(Age, WebPages_{\Wk} | Sex=s_fixed, Wk=wk_fixed)
def make_base_proposal_H_SWk(
    static_fixed_sex_idx: int, static_target_page_idx: int, static_num_web_pages: int
):
    @gen
    def base_proposal_H_SWk_specialized(
        p_age_vector: jnp.ndarray,  # p(Age)
        web_pages_cpt: jnp.ndarray,  # CPT P(WebPage_i | Sex, Age)
    ):
        age_idx = categorical(probs=p_age_vector) @ "age_q0_SWk"
        other_webpages_states_list = []
        # Probabilities for ALL web pages given fixed sex_idx and sampled age_idx
        prob_watch_vector_all = jnp.clip(
            web_pages_cpt[:static_num_web_pages, static_fixed_sex_idx, age_idx],
            1e-7,
            1.0 - 1e-7,
        )
        for i in range(static_num_web_pages):
            if i == static_target_page_idx:  # Wk is fixed
                continue
            # Sample other web pages
            page_watched = (
                bernoulli(probs=prob_watch_vector_all[i]) @ f"web_page_q0_SWk_{i}"
            )
            other_webpages_states_list.append(page_watched)
        return age_idx, (  # Returns (age_idx, other_webpages_array)
            jnp.stack(other_webpages_states_list)
            if other_webpages_states_list
            else jnp.array([])
        )

    return base_proposal_H_SWk_specialized


# --- Helper functions for generic EEVI estimator ---


# reconstruct_z_values_fn: (x_sample_from_q0, y_sample) -> z_values (tuple for BN)
# These are the existing reconstruct_z_H_S, reconstruct_z_H_Wk, reconstruct_z_H_SWk
@jit
def reconstruct_z_H_S_values(  # Renamed to avoid conflict if old one is kept temporarily
    x_sample: tuple[FloatArray, FloatArray],
    y_sample_sex: FloatArray,  # x_sample is (age_idx_q0, webpages_q0)
) -> tuple:  # Returns (sex_val, age_val, webpages_val)
    age_idx_q0, webpages_q0 = x_sample
    return (y_sample_sex.astype(jnp.int32), age_idx_q0.astype(jnp.int32), webpages_q0)


@partial(jit, static_argnums=(2,))
def reconstruct_z_H_Wk_values(  # Renamed
    x_sample: tuple[
        FloatArray, FloatArray, FloatArray
    ],  # x_sample is (sex_idx_q0, age_idx_q0, other_pages_q0)
    y_sample_wk: FloatArray,  # wk_val_y
    target_page_idx: int,
) -> tuple:  # Returns (sex_val, age_val, all_webpages_val)
    sex_idx_q0, age_idx_q0, other_pages_q0 = x_sample
    all_webpages = jnp.zeros(
        num_web_pages_global, dtype=jnp.int32
    )  # num_web_pages_global needs to be accessible
    current_other_idx = 0
    for i in range(num_web_pages_global):
        if i == target_page_idx:
            all_webpages = all_webpages.at[i].set(y_sample_wk.astype(jnp.int32))
        else:
            # Ensure other_pages_q0 is not empty and index is within bounds
            if other_pages_q0.size > 0 and current_other_idx < other_pages_q0.shape[0]:
                all_webpages = all_webpages.at[i].set(
                    other_pages_q0[current_other_idx].astype(jnp.int32)
                )
                current_other_idx += 1
            # else: page remains 0 if not in other_pages_q0 (e.g. if num_web_pages=1 and target is that page)
    return (sex_idx_q0.astype(jnp.int32), age_idx_q0.astype(jnp.int32), all_webpages)


@partial(jit, static_argnums=(2,))
def reconstruct_z_H_SWk_values(  # Renamed
    x_sample: tuple[FloatArray, FloatArray],  # x_sample is (age_idx_q0, other_pages_q0)
    y_sample_s_wk: tuple[FloatArray, FloatArray],  # (sex_val_y, wk_val_y)
    target_page_idx: int,
) -> tuple:  # Returns (sex_val, age_val, all_webpages_val)
    age_idx_q0, other_pages_q0 = x_sample
    sex_val_y, wk_val_y = y_sample_s_wk
    all_webpages = jnp.zeros(
        num_web_pages_global, dtype=jnp.int32
    )  # num_web_pages_global
    current_other_idx = 0
    for i in range(num_web_pages_global):
        if i == target_page_idx:
            all_webpages = all_webpages.at[i].set(wk_val_y.astype(jnp.int32))
        else:
            if other_pages_q0.size > 0 and current_other_idx < other_pages_q0.shape[0]:
                all_webpages = all_webpages.at[i].set(
                    other_pages_q0[current_other_idx].astype(jnp.int32)
                )
                current_other_idx += 1
    return (sex_val_y.astype(jnp.int32), age_idx_q0.astype(jnp.int32), all_webpages)


# build_p_choices_fn: (z_values, num_web_pages_for_p_assess) -> ChoiceMap for p_model (full_bn_model)
def build_p_choices_for_bn(
    z_values: tuple, num_web_pages_for_p_assess: int
) -> genjax.ChoiceMap:
    sex_val, age_val, webpages_val = z_values
    choices = C.n()
    choices = choices | C["sex"].set(sex_val.astype(jnp.int32))
    choices = choices | C["age"].set(age_val.astype(jnp.int32))
    for i_wp in range(num_web_pages_for_p_assess):
        # Ensure webpages_val has enough elements
        if i_wp < webpages_val.shape[0]:
            choices = choices | C[f"web_page_{i_wp}"].set(
                webpages_val[i_wp].astype(jnp.int32)
            )
        # else: if webpages_val is shorter, this might be an issue or imply those pages are not set.
        # Assuming webpages_val always has at least num_web_pages_for_p_assess elements.
    return choices


# extract_y_prime_fn: (joint_choice_from_p) -> y_prime_sample
# These are the existing curried_extract_y_prime_H_S, ..._H_Wk, ..._H_SWk

# extract_x_prime_values_fn: (joint_choice_from_p) -> x_prime_values (tuple for BN q0 models)
# These are the existing curried_extract_x_prime_H_S, ..._H_Wk, ..._H_SWk


# build_q0_choices_from_x_prime_fn: (x_prime_values, specific_args_for_q0_type) -> ChoiceMap
def build_q0_H_S_choices(x_prime_values: tuple) -> genjax.ChoiceMap:
    age_val_xp, webpages_val_xp = (
        x_prime_values  # x_prime from q0_H_S is (age, webpages)
    )
    choices = C.n()
    choices = choices | C["age_q0_S"].set(age_val_xp.astype(jnp.int32))
    for i, wp_val in enumerate(webpages_val_xp):
        choices = choices | C[f"web_page_q0_S_{i}"].set(wp_val.astype(jnp.int32))
    return choices


def build_q0_H_Wk_choices(
    x_prime_values: tuple, target_page_idx: int, num_web_pages: int
) -> genjax.ChoiceMap:
    sex_val_xp, age_val_xp, other_wps_xp = (
        x_prime_values  # x_prime from q0_H_Wk is (sex, age, other_webpages)
    )
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
    age_val_xp, other_wps_xp = (
        x_prime_values  # x_prime from q0_H_SWk is (age, other_webpages)
    )
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


# sample_y_fn: (key, n_samples, y_var_type, p_sex, p_age, web_cpt, target_idx, n_sex, n_age) -> y_samples
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
) -> tuple[FloatArray, ...] | FloatArray:  # Returns tuple for SWk, else array
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
        # Ensure there are 2 categories for Wk for floor division by 2 to work as intended for s_indices
        num_wk_categories = p_s_wk_joint.shape[1]  # Should be 2 (watched/not watched)
        sampled_indices_flat = random.categorical(
            key, logits=jnp.log(jnp.clip(p_s_wk_flat, 1e-9)), shape=(n_samples,)
        )
        s_indices = sampled_indices_flat // num_wk_categories
        wk_states = sampled_indices_flat % num_wk_categories
        return (s_indices, wk_states)  # Returns a tuple of arrays
    raise ValueError(f"Unknown y_var_type: {y_var_type}")


# --- Original curried extract functions (can be used directly) ---
def curried_extract_y_prime_H_S(joint_choice: genjax.ChoiceMap):
    return joint_choice["sex"]


def curried_extract_x_prime_H_S(joint_choice: genjax.ChoiceMap, num_web_pages: int):
    age_val = joint_choice["age"]
    webpages_val = jnp.array(
        [joint_choice[f"web_page_{i}"] for i in range(num_web_pages)]
    )
    return (age_val, webpages_val)


def curried_extract_y_prime_H_Wk(joint_choice: genjax.ChoiceMap, target_page_idx: int):
    return joint_choice[f"web_page_{target_page_idx}"]


def curried_extract_x_prime_H_Wk(
    joint_choice: genjax.ChoiceMap, target_page_idx: int, num_web_pages: int
):
    sex_val = joint_choice["sex"]
    age_val = joint_choice["age"]
    other_pages_list = [
        joint_choice[f"web_page_{i}"]
        for i in range(num_web_pages)
        if i != target_page_idx
    ]
    return (
        sex_val,
        age_val,
        jnp.stack(other_pages_list) if other_pages_list else jnp.array([]),
    )


def curried_extract_y_prime_H_SWk(joint_choice: genjax.ChoiceMap, target_page_idx: int):
    return (joint_choice["sex"], joint_choice[f"web_page_{target_page_idx}"])


def curried_extract_x_prime_H_SWk(
    joint_choice: genjax.ChoiceMap, target_page_idx: int, num_web_pages: int
):
    age_val = joint_choice["age"]
    other_pages_list = [
        joint_choice[f"web_page_{i}"]
        for i in range(num_web_pages)
        if i != target_page_idx
    ]
    return (age_val, jnp.stack(other_pages_list) if other_pages_list else jnp.array([]))


# --- Pytest Test Function (エラー処理を省略) ---
def test_mutual_information_estimation_with_e_evi():
    # print("\n*********** データの前処理 ***********")
    (
        sex_map_gl,
        age_map_gl,
        web_cpt_gl,
        p_sex_gl,
        p_age_gl,
        n_sex_gl,
        n_age_gl,
    ) = preprocess_data_and_calculate_params(df_global)

    # print(f"Sex categories: {n_sex_gl}, Age categories: {n_age_gl}, Web pages: {num_web_pages_global}")
    p_sex_gl = jnp.clip(p_sex_gl, 1e-9, 1.0)
    p_age_gl = jnp.clip(p_age_gl, 1e-9, 1.0)
    web_cpt_gl = jnp.clip(web_cpt_gl, 1e-7, 1.0 - 1e-7)
    print(f"P(Sex) from data = {p_sex_gl}")
    print(f"P(Age) from data = {p_age_gl}")

    TARGET_PAGE_FOR_MI = 0
    # print(f"\n*********** Calculating MI(Sex; WebPage_{TARGET_PAGE_FOR_MI}) ***********")

    print("\n--- Analytical MI ---")
    analytical_mi_val, an_H_S, an_H_Wk, an_H_SWk = calculate_analytical_mi(
        p_sex_gl, p_age_gl, web_cpt_gl, TARGET_PAGE_FOR_MI, n_sex_gl, n_age_gl
    )
    print(f"  Analytical H(S)      = {an_H_S:.4f} bits")
    print(f"  Analytical H(W{TARGET_PAGE_FOR_MI})    = {an_H_Wk:.4f} bits")
    print(f"  Analytical H(S,W{TARGET_PAGE_FOR_MI}) = {an_H_SWk:.4f} bits")
    print(f"  Analytical MI(S;W{TARGET_PAGE_FOR_MI}) = {analytical_mi_val:.4f} bits")

    print("\n--- EEVI MI ---")
    N_OUTER = 1000
    P_SIR = 100
    MAIN_KEY = random.PRNGKey(123)
    p_model_eevi_final = full_bn_model  # This is p(Z)

    # Common arguments for p_model
    p_model_args_eevi = (p_sex_gl, p_age_gl, web_cpt_gl)
    # Common function to build p_model choices from z_values
    build_p_choices_fn_curried = functools.partial(
        build_p_choices_for_bn, num_web_pages_for_p_assess=num_web_pages_global
    )

    # --- Estimate H(S) ---
    key_hs, MAIN_KEY = random.split(MAIN_KEY)
    print(f"Estimating H(S) with EEVI (N_outer={N_OUTER}, P_sir={P_SIR})...")
    FIXED_SEX_FOR_Q0_HS = 0  # Example: condition q0 on Sex=0
    q0_model_H_S_spec = make_base_proposal_H_S(
        static_fixed_sex_idx=FIXED_SEX_FOR_Q0_HS,
        static_num_web_pages=num_web_pages_global,
    )
    q0_args_H_S = (p_age_gl, web_cpt_gl)  # Args for q0_H_S: p(Age), CPT

    # Specific functions for H(S) estimation
    reconstruct_z_fn_H_S = reconstruct_z_H_S_values
    extract_y_prime_fn_H_S = curried_extract_y_prime_H_S
    extract_x_prime_values_fn_H_S = functools.partial(
        curried_extract_x_prime_H_S, num_web_pages=num_web_pages_global
    )
    build_q0_choices_fn_H_S = build_q0_H_S_choices

    y_sampling_args_H_S = (
        "S",
        p_sex_gl,
        p_age_gl,
        web_cpt_gl,
        -1,
        n_sex_gl,
        n_age_gl,
    )  # target_page_idx=-1 as not used for H(S)
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
        static_target_page_idx=TARGET_PAGE_FOR_MI,
        static_num_web_pages=num_web_pages_global,
    )
    q0_args_H_Wk = (
        p_sex_gl,
        p_age_gl,
        web_cpt_gl,
    )  # Args for q0_H_Wk: p(Sex), p(Age), CPT

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
    FIXED_SEX_FOR_Q0_HSWK = 0  # Example
    q0_model_H_SWk_spec = make_base_proposal_H_SWk(
        static_fixed_sex_idx=FIXED_SEX_FOR_Q0_HSWK,
        static_target_page_idx=TARGET_PAGE_FOR_MI,
        static_num_web_pages=num_web_pages_global,
    )
    q0_args_H_SWk = (p_age_gl, web_cpt_gl)  # Args for q0_H_SWk: p(Age), CPT

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

    tilde_MI_eeri = tilde_H_S + tilde_H_Wk - hat_H_SWk
    hat_MI_eeri = hat_H_S + hat_H_Wk - tilde_H_SWk
    print(
        f"\n  EEVI MI(S;W{TARGET_PAGE_FOR_MI}) Interval: [{tilde_MI_eeri:.4f}, {hat_MI_eeri:.4f}] bits"
    )
    print(
        f"  Analytical MI(S;W{TARGET_PAGE_FOR_MI})      : {analytical_mi_val:.4f} bits"
    )

    tolerance = 0.20
    assert (
        tilde_MI_eeri - tolerance
    ) <= analytical_mi_val, f"Analytical MI {analytical_mi_val} is less than lower EEVI bound {tilde_MI_eeri} (with tolerance {tolerance})"
    assert analytical_mi_val <= (
        hat_MI_eeri + tolerance
    ), f"Analytical MI {analytical_mi_val} is greater than upper EEVI bound {hat_MI_eeri} (with tolerance {tolerance})"
