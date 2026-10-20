import json
from functools import lru_cache
import numpy as np
import pandas as pd
from app.agents.ollama_agent import ask_json, llm_enabled
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer

def _safe_k(y):
    return max(1, min(5, int(y.value_counts().min())-1))


@lru_cache(maxsize=64)
def _cached_llm_category_proposals(profile_json: str) -> str | None:
    """Cache LLM category suggestions so the same dataset isn't sent on every candidate."""
    profile = json.loads(profile_json)
    proposal = ask_json(
        "You propose cautious, meaningful new category values for tabular data augmentation. "
        "Use only patterns or combinations supported by observed values; do not invent factual claims. "
        "For each column propose zero to three unseen values. If the domain is ambiguous, return an empty list. "
        "Never change target labels. Return JSON shaped as {categories: {column_name: [new_value, ...]}}.",
        {"categorical_profile": profile}, num_predict=300,
    )
    return json.dumps(proposal, ensure_ascii=False) if proposal else None

def _categorical_candidates(series):
    """Infer conservative, unseen category candidates from patterns already in a column.

    Supported patterns:
    - Numeric intervals such as ``0-2 years`` -> ``6-8 years`` when interval steps agree.
    - Numbered labels such as ``Level 1``, ``Level 2`` -> ``Level 3``.
    - Two-part labels such as ``Data Analyst`` and ``Data Scientist`` -> ``Data Scientist``-like
      combinations of already observed components (e.g. ``Data Engineer``), without inventing
      new component words.

    These are candidates, not guaranteed domain truths; metadata makes generated values auditable.
    """
    import re

    values = [str(v).strip() for v in series.dropna().unique() if str(v).strip()]
    existing = set(values)
    candidates = []

    # Extrapolate consistently spaced numeric ranges while preserving prefix/suffix formatting.
    range_rows = []
    range_re = re.compile(r"^(.*?)(\d+)\s*([-–])\s*(\d+)(.*)$")
    for value in values:
        match = range_re.match(value)
        if match:
            prefix, lo, sep, hi, suffix = match.groups()
            range_rows.append((value, prefix, int(lo), sep, int(hi), suffix))
    if len(range_rows) >= 2:
        range_rows.sort(key=lambda row: row[2])
        lows = [row[2] for row in range_rows]
        steps = [b - a for a, b in zip(lows, lows[1:]) if b > a]
        widths = [row[4] - row[2] for row in range_rows]
        if steps and len(set(steps)) == 1 and len(set(widths)) == 1:
            _, prefix, _, sep, _, suffix = range_rows[-1]
            step, width = steps[0], widths[-1]
            candidate = f"{prefix}{lows[-1] + step}{sep}{lows[-1] + step + width}{suffix}"
            if candidate not in existing:
                candidates.append(candidate)

    # Extend numbered categories only if at least two values share an identical template.
    numbered = {}
    number_re = re.compile(r"^(.*?)(\d+)(.*?)$")
    for value in values:
        match = number_re.match(value)
        if match:
            prefix, number, suffix = match.groups()
            numbered.setdefault((prefix, suffix), []).append(int(number))
    for (prefix, suffix), nums in numbered.items():
        if len(set(nums)) >= 2 and len(candidates) < 4:
            candidate = f"{prefix}{max(nums) + 1}{suffix}"
            if candidate not in existing:
                candidates.append(candidate)

    # Conservatively recombine observed tokens in two-part labels. Only use this when the
    # column clearly contains repeated first components and at least three composite values.
    composite_rows = []
    for value in values:
        match = re.match(r"^([A-Za-z][A-Za-z0-9&']*)\s+([A-Za-z][A-Za-z0-9&']*)$", value)
        if match:
            composite_rows.append((match.group(1), match.group(2)))
    if len(composite_rows) >= 3:
        firsts = [a for a, _ in composite_rows]
        seconds = [b for _, b in composite_rows]
        repeated_firsts = {token for token in firsts if firsts.count(token) >= 2}
        for first in sorted(repeated_firsts):
            for second in sorted(set(seconds)):
                candidate = f"{first} {second}"
                if candidate not in existing and candidate not in candidates:
                    candidates.append(candidate)
                    if len(candidates) >= 8:
                        break
            if len(candidates) >= 8:
                break

    return candidates


def _similarity_oversample(
    train_df, target_column, max_generated_fraction=0.20, random_state=42,
    allow_new_categories=True,
):
    """Create bounded, local synthetic rows from same-class parent records.

    Numeric values are interpolated between two similar same-class records. Categorical features
    usually reuse parent values; when enabled, a small share can use an inferred unseen category
    based on patterns/components already present in that column. The target label is never
    synthesized or changed. Original rows remain unchanged and first in the result.
    """
    rng = np.random.default_rng(random_state)
    train_df = train_df.reset_index(drop=True).copy()
    feature_cols = [c for c in train_df.columns if c != target_column]
    if not feature_cols or len(train_df) < 2:
        return train_df.copy(), 0, {}, {}

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    class_counts = train_df[target_column].value_counts()
    if len(class_counts) < 1:
        return train_df.copy(), 0, {}, {}

    generated_budget = max(1, int(round(len(train_df) * max_generated_fraction)))
    generated_budget = min(generated_budget, max(1, len(train_df)))
    imbalanced = class_counts.max() / max(1, class_counts.min()) >= 1.5
    candidate_classes = (
        list(class_counts[class_counts < class_counts.max()].index) if imbalanced
        else list(class_counts.index)
    ) or list(class_counts.index)

    numeric_frame = train_df[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=train_df.index)
    if numeric_cols:
        medians = numeric_frame.median().fillna(0)
        numeric_frame = numeric_frame.fillna(medians)
        scales = (numeric_frame.quantile(.75) - numeric_frame.quantile(.25)).replace(0, 1).fillna(1)
        numeric_frame = (numeric_frame - numeric_frame.median()) / scales
    categorical_frame = train_df[categorical_cols].astype("string").fillna("<MISSING>") if categorical_cols else pd.DataFrame(index=train_df.index)
    # Convert once to NumPy arrays. Repeated pandas .loc calls inside the neighbor search
    # were the main reason a single augmentation candidate could take several seconds.
    numeric_values = numeric_frame.to_numpy(dtype=float) if numeric_cols else np.empty((len(train_df), 0))
    categorical_values = categorical_frame.to_numpy(dtype=object) if categorical_cols else np.empty((len(train_df), 0), dtype=object)

    def distances_from(anchor_idx, candidate_indices):
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        total = np.zeros(len(candidate_indices), dtype=float)
        dimensions = 0
        if numeric_cols:
            delta = numeric_values[candidate_indices] - numeric_values[anchor_idx]
            total += np.mean(delta * delta, axis=1)
            dimensions += 1
        if categorical_cols:
            mismatches = categorical_values[candidate_indices] != categorical_values[anchor_idx]
            total += np.mean(mismatches, axis=1)
            dimensions += 1
        return total / max(1, dimensions)

    category_candidates = {
        col: _categorical_candidates(train_df[col]) for col in categorical_cols
    } if allow_new_categories else {}
    llm_category_candidates = {}
    if allow_new_categories and categorical_cols and llm_enabled():
        # Only unique values/counts are sent to the local LLM; no raw row records or target labels.
        category_profile = {}
        for col in categorical_cols[:30]:
            series = train_df[col].dropna().astype(str)
            counts = series.value_counts().head(30)
            category_profile[str(col)] = {
                "observed_values": [str(v)[:100] for v in counts.index.tolist()],
                "counts": {str(k)[:100]: int(v) for k, v in counts.items()},
            }
        cached_proposal = _cached_llm_category_proposals(json.dumps(category_profile, sort_keys=True, ensure_ascii=False))
        try:
            proposal = json.loads(cached_proposal) if cached_proposal else None
        except (json.JSONDecodeError, TypeError):
            proposal = None
        raw_categories = proposal.get("categories", {}) if isinstance(proposal, dict) else {}
        if isinstance(raw_categories, dict):
            for col in categorical_cols:
                raw = raw_categories.get(str(col), [])
                if not isinstance(raw, list):
                    continue
                observed = set(train_df[col].dropna().astype(str).str.strip())
                valid = []
                for value in raw[:3]:
                    if not isinstance(value, (str, int, float)):
                        continue
                    candidate = str(value).strip()
                    if candidate and len(candidate) <= 100 and candidate not in observed and candidate not in valid:
                        valid.append(candidate)
                if valid:
                    llm_category_candidates[col] = valid
                    category_candidates[col] = list(dict.fromkeys(category_candidates.get(col, []) + valid))
    generated_category_values = {col: set() for col in categorical_cols}
    rows = []
    class_weights = np.array([1.0 / max(1, int(class_counts[c])) for c in candidate_classes], dtype=float)
    class_weights /= class_weights.sum()
    for _ in range(generated_budget):
        cls = rng.choice(candidate_classes, p=class_weights)
        class_rows = train_df.index[train_df[target_column] == cls].tolist()
        if not class_rows:
            continue
        anchor_idx = class_rows[int(rng.integers(0, len(class_rows)))]
        if len(class_rows) == 1:
            neighbor_idx = anchor_idx
        else:
            neighbor_indices = [j for j in class_rows if j != anchor_idx]
            distance_values = distances_from(anchor_idx, neighbor_indices)
            nearest_order = np.argsort(distance_values)[:min(3, len(neighbor_indices))]
            neighbor_pool = [neighbor_indices[int(pos)] for pos in nearest_order]
            neighbor_idx = neighbor_pool[int(rng.integers(0, len(neighbor_pool)))]
        left, right = train_df.loc[anchor_idx], train_df.loc[neighbor_idx]
        alpha = float(rng.uniform(0.15, 0.85)) if anchor_idx != neighbor_idx else 0.0
        row = {}
        for col in feature_cols:
            a, b = left[col], right[col]
            if col in numeric_cols:
                try:
                    if pd.isna(a): value = b
                    elif pd.isna(b): value = a
                    else:
                        value = float(a) + alpha * (float(b) - float(a))
                        value = min(max(value, min(float(a), float(b))), max(float(a), float(b)))
                    if pd.api.types.is_integer_dtype(train_df[col].dtype) and not pd.isna(value):
                        value = int(round(value))
                except (TypeError, ValueError):
                    value = a if rng.random() < 0.5 else b
            else:
                candidates = category_candidates.get(col, [])
                # Limit novel categories to a minority of synthetic values. Use only candidates
                # inferred from this column; otherwise fall back to a value from a parent row.
                can_generate = candidates and (rng.random() < 0.20 or (not generated_category_values[col] and len(rows) == 0))
                if can_generate:
                    unused = [v for v in candidates if v not in generated_category_values[col]]
                    if unused:
                        value = unused[int(rng.integers(0, len(unused)))]
                        generated_category_values[col].add(value)
                    else:
                        value = a if rng.random() < 0.5 else b
                else:
                    value = a if rng.random() < 0.5 else b
            row[col] = value
        row[target_column] = cls
        rows.append(row)

    if not rows:
        return train_df.copy(), 0, {}, {}
    synthetic = pd.DataFrame(rows, columns=train_df.columns)
    for col in train_df.columns:
        try:
            synthetic[col] = synthetic[col].astype(train_df[col].dtype)
        except (TypeError, ValueError):
            pass
    out = pd.concat([train_df.copy(), synthetic], ignore_index=True)
    generated_values = {
        col: sorted(values) for col, values in generated_category_values.items() if values
    }
    return out, len(synthetic), generated_values, llm_category_candidates


def augment_training_data(train_df, target_column, strategy):
    train_df = train_df.copy()
    feature_cols = [c for c in train_df.columns if c != target_column]
    if not feature_cols: raise ValueError("At least one feature column is required.")
    X, y = train_df[feature_cols].copy(), train_df[target_column].copy()
    empty_cols = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=empty_cols)
    feature_cols = list(X.columns)
    if not feature_cols: raise ValueError("All feature columns are empty.")
    cat = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in feature_cols if c not in cat]
    for c in num: X[c] = pd.to_numeric(X[c], errors="coerce")
    if num: X[num] = SimpleImputer(strategy="median").fit_transform(X[num])
    if cat:
        X[cat] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat])
        for c in cat: X[c] = X[c].astype(str)
    method = strategy["strategy"]
    if method == "SimilarityOverSampler":
        out, generated, generated_category_values, llm_category_candidates = _similarity_oversample(
            train_df, target_column,
            allow_new_categories=bool(strategy.get("allow_new_categories", True)),
        )
        out["__autodatagen_source"] = ["original"] * len(train_df) + ["synthetic_similar"] * generated
        before = y.value_counts()
        after = out[target_column].value_counts()
        metadata = {
            "requested_strategy": method, "used_strategy": method,
            "original_training_rows": int(len(train_df)), "augmented_training_rows": int(len(out)),
            "generated_rows": int(generated),
            "generated_fraction": round(generated / max(1, len(train_df)), 4),
            "generation_method": "same-class local interpolation for numeric features; categorical features mostly copied from parent rows, with optional conservative unseen-category synthesis from observed value patterns",
            "allow_new_categories": bool(strategy.get("allow_new_categories", True)),
            "generated_new_categorical_values": generated_category_values,
            "llm_proposed_categorical_candidates": llm_category_candidates,
            "llm_category_generation_enabled": bool(strategy.get("allow_new_categories", True) and llm_enabled()),
            "class_counts_before": {str(k): int(v) for k, v in before.items()},
            "class_counts_after": {str(k): int(v) for k, v in after.items()},
            "note": "Unseen categorical candidates are inferred from patterns/components already observed in each column and may still require domain validation. Synthetic rows are candidates only; the orchestrator must reject candidates that fail evaluation safeguards."
        }
        return out, metadata
    if method in {"NoResampling", "ClassWeight"}:
        out = train_df.copy()
        out["__autodatagen_source"] = "original"
        counts = y.value_counts()
        metadata = {
            "requested_strategy": method, "used_strategy": method,
            "original_training_rows": int(len(train_df)),
            "augmented_training_rows": int(len(out)), "generated_rows": 0,
            "class_counts_before": {str(k): int(v) for k, v in counts.items()},
            "class_counts_after": {str(k): int(v) for k, v in counts.items()},
            "note": "No rows were resampled. ClassWeight changes model fitting weights, not the exported dataset distribution."
        }
        return out, metadata
    if method == "RandomUnderSampler":
        sampler = RandomUnderSampler(random_state=42)
    elif method == "TomekLinks" and not cat:
        sampler = TomekLinks()
    elif method == "SMOTETomek" and not cat:
        sampler = SMOTETomek(smote=SMOTE(random_state=42, k_neighbors=_safe_k(y)), random_state=42)
    elif method == "SMOTENC" and cat:
        sampler = SMOTENC(categorical_features=[feature_cols.index(c) for c in cat],
                          random_state=42, k_neighbors=_safe_k(y))
    elif method == "SMOTE" and not cat:
        sampler = SMOTE(random_state=42, k_neighbors=_safe_k(y))
    elif method == "RandomOverSampler":
        sampler = RandomOverSampler(random_state=42)
    else:
        raise ValueError(f"Strategy {method} is incompatible with these feature types.")
    try:
        xr, yr = sampler.fit_resample(X, y)
        used = sampler.__class__.__name__
    except Exception:
        sampler = RandomOverSampler(random_state=42)
        xr, yr = sampler.fit_resample(X, y)
        used = "RandomOverSampler (fallback)"
    out = pd.DataFrame(xr, columns=feature_cols)
    out[target_column] = np.asarray(yr)
    for c in empty_cols: out[c] = np.nan
    out = out[list(train_df.columns)]
    out["__autodatagen_source"] = ["original" if i < len(train_df) else "synthetic_or_resampled" for i in range(len(out))]
    metadata = {"requested_strategy":method, "used_strategy":used,
        "original_training_rows":int(len(train_df)), "augmented_training_rows":int(len(out)),
        "generated_rows":int(max(0,len(out)-len(train_df))),
        "class_counts_before":{str(k):int(v) for k,v in y.value_counts().items()},
        "class_counts_after":{str(k):int(v) for k,v in pd.Series(yr).value_counts().items()},
        "note":"SMOTE/SMOTENC interpolate samples; RandomOverSampler duplicates minority rows. Neither guarantees privacy or realism."}
    return out, metadata
