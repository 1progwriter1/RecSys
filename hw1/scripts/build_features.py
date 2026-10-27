import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


def create_id_tables(history, catalog):
    users = pd.DataFrame({"user_id": sorted(history["user_id"].unique())})
    users["user_idx"] = np.arange(len(users), dtype=np.int32)
    items = pd.DataFrame({"item_id": sorted(history["item_id"].unique())})
    items["item_idx"] = np.arange(len(items), dtype=np.int32)
    items = items.merge(catalog, on="item_id", validate="one_to_one", how="left")
    return users, items


def add_indices(data, users, items, allow_unknown=False):
    user_map = users.set_index("user_id")["user_idx"]
    item_map = items.set_index("item_id")["item_idx"]
    dtype = "Int32" if allow_unknown else "int32"
    return data.assign(
        user_idx=data["user_id"].map(user_map).astype(dtype),
        item_idx=data["item_id"].map(item_map).astype(dtype),
    )


def prepare_evaluation(data, users, items):
    evaluation = add_indices(data, users, items, allow_unknown=True)
    evaluation["eligible"] = evaluation[["user_idx", "item_idx"]].notna().all(axis=1)
    ground_truth = evaluation[evaluation["eligible"] & evaluation["is_relevant"]].copy()
    return evaluation, ground_truth


def create_interaction_matrices(history, shape):
    indices = (history["user_idx"].to_numpy(), history["item_idx"].to_numpy())
    seen = sparse.csr_matrix(
        (np.ones(len(history), dtype=np.float32), indices), shape=shape,
    )
    positive = history[history["is_relevant"]]
    positive_indices = (positive["user_idx"].to_numpy(), positive["item_idx"].to_numpy())
    positives = sparse.csr_matrix(
        (np.ones(len(positive), dtype=np.float32), positive_indices), shape=shape,
    )
    return {"seen": seen, "positives": positives}


def create_item_features(items):
    brand_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    brand_features = brand_encoder.fit_transform(items[["brand"]])
    category_encoder = MultiLabelBinarizer(sparse_output=True)
    category_features = category_encoder.fit_transform(items["categories"])
    features = sparse.hstack(
        [brand_features, category_features], format="csr", dtype=np.float32,
    )
    names = brand_encoder.get_feature_names_out(["brand"]).tolist()
    names += [f"category={category}" for category in category_encoder.classes_]
    return features, names


def validate_stage(history, evaluation, matrices):
    if matrices["seen"].nnz != len(history):
        raise ValueError("Repeated user-item pairs were merged in the sparse matrix")
    if history["timestamp"].max() >= evaluation["timestamp"].min():
        raise ValueError("The history overlaps the evaluation period")


def build_stage_report(tables, feature_names):
    history = tables["history"]
    evaluation = tables["evaluation"]
    ground_truth = tables["ground_truth"]
    return {
        "history_rows": len(history),
        "users": len(tables["users"]),
        "candidate_items": len(tables["items"]),
        "evaluation_rows": len(evaluation),
        "warm_evaluation_rows": int(evaluation["eligible"].sum()),
        "excluded_cold_rows": int((~evaluation["eligible"]).sum()),
        "cold_user_rows": int(evaluation["user_idx"].isna().sum()),
        "cold_item_rows": int(evaluation["item_idx"].isna().sum()),
        "cold_user_and_item_rows": int(
            evaluation[["user_idx", "item_idx"]].isna().all(axis=1).sum()
        ),
        "ground_truth_rows": len(ground_truth),
        "ranking_users": int(ground_truth["user_id"].nunique()),
        "item_feature_count": len(feature_names),
    }


def build_stage(history, evaluation, catalog):
    users, items = create_id_tables(history, catalog)
    history = add_indices(history, users, items)
    evaluation, ground_truth = prepare_evaluation(evaluation, users, items)
    tables = {
        "users": users,
        "items": items,
        "history": history,
        "evaluation": evaluation,
        "ground_truth": ground_truth,
    }

    matrices = create_interaction_matrices(history, shape=(len(users), len(items)))
    matrices["item_features"], feature_names = create_item_features(items)
    validate_stage(history, evaluation, matrices)

    report = build_stage_report(tables, feature_names)
    return {"history": history, "evaluation": evaluation, "ground_truth": ground_truth,
            **matrices}, report


def create_evaluation_stages(frames):
    final_history = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
    return {
        "validation": (frames["train"], frames["validation"]),
        "test": (final_history, frames["test"]),
    }
