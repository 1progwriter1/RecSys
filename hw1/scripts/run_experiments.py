import gc
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from models import EASE, FM, IALS, ItemKNN, TimeSVDPlusPlus

SEED = 42
K = 10
SEARCH_SPACE = {
    "ItemKNN": [{"neighbors": k, "shrinkage": 10.0} for k in [50, 150]],
    "iALS": [{"factors": 32, "regularization": reg, "alpha": 20.0,
              "iterations": 10, "seed": SEED} for reg in [1.0, 10.0]],
    "EASE": [{"regularization": reg} for reg in [100.0, 500.0]],
    "FM": [{"factors": 16, "regularization": 1e-5, "learning_rate": 0.01,
            "epochs": epochs, "batch_size": 8192, "use_metadata": True, "seed": SEED}
           for epochs in [10, 20]],
    "timeSVD++": [{"factors": 16, "regularization": 20.0, "learning_rate": rate,
                   "epochs": 50, "bin_days": 90, "beta": 0.4, "seed": SEED}
                  for rate in [0.01, 0.03]],
}
MODELS = {"ItemKNN": ItemKNN, "iALS": IALS, "EASE": EASE, "FM": FM,
          "FM (IDs only)": FM, "timeSVD++": TimeSVDPlusPlus}


def top_k(scores, k):
    available = np.flatnonzero(np.isfinite(scores))
    if len(available) <= k:
        return available[np.lexsort((available, -scores[available]))]
    threshold = np.partition(scores[available], len(available) - k)[-k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    chosen = np.concatenate([above, tied[:k - len(above)]])
    return chosen[np.lexsort((chosen, -scores[chosen]))]


def evaluate(model, data, k=K, batch_size=64):
    truth = data["ground_truth"].groupby("user_idx")["item_idx"].agg(set).to_dict()
    users = np.array(sorted(truth), dtype=np.int32)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    recalls, ndcgs = [], []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.array(model.score(batch), copy=True)
        if scores.shape != (len(batch), data["seen"].shape[1]) or not np.isfinite(scores).all():
            raise ValueError("Invalid candidate scores")
        for row, user in enumerate(batch):
            left, right = data["seen"].indptr[user:user + 2]
            seen = data["seen"].indices[left:right]
            scores[row, seen] = -np.inf
            selected = top_k(scores[row], k)
            if np.intersect1d(selected, seen).size or len(selected) != len(set(selected)):
                raise ValueError("Invalid recommendation list")
            hits = np.array([item in truth[user] for item in selected], dtype=float)
            recalls.append(hits.sum() / len(truth[user]))
            ndcgs.append((hits * discounts[:len(selected)]).sum()
                         / discounts[:min(k, len(truth[user]))].sum())
    return {f"Recall@{k}": float(np.mean(recalls)), f"NDCG@{k}": float(np.mean(ndcgs))}


def run_trial(stage, data, name, parameters):
    print(f"{stage} {name} {parameters}: fitting", flush=True)
    model = MODELS[name](**parameters).fit(data)
    result = {"stage": stage, "model": name,
              "parameters": json.dumps(parameters, sort_keys=True), **evaluate(model, data)}
    print(f"{stage} {name}: NDCG@10={result['NDCG@10']:.6f}, "
          f"Recall@10={result['Recall@10']:.6f}", flush=True)
    del model
    gc.collect()
    return result


def run_experiments(stages):
    with threadpool_limits(limits=4):
        validation = stages["validation"]
        trials, selected = [], {}
        for name, configurations in SEARCH_SPACE.items():
            candidates = [run_trial("validation", validation, name, params)
                          for params in configurations]
            trials.extend(candidates)
            best = max(candidates, key=lambda row: row["NDCG@10"])
            selected[name] = json.loads(best["parameters"])
        ablation_trials = [run_trial("validation", validation, "FM (IDs only)",
                                    {**params, "use_metadata": False})
                           for params in SEARCH_SPACE["FM"]]
        trials.extend(ablation_trials)
        selected["FM (IDs only)"] = json.loads(
            max(ablation_trials, key=lambda row: row["NDCG@10"])["parameters"]
        )
        test = stages["test"]
        final = [run_trial("test", test, name, params) for name, params in selected.items()]
    return pd.DataFrame(trials), pd.DataFrame(final)
