import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


FEATURE_COLUMNS = [
    "trend_score",
    "level_score",
    "distance_score",
    "impulse_score",
    "structure_score",
]


def parse_note(value):
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_journal(path="journal.csv"):
    df = pd.read_csv(path)
    meta = df["note"].apply(parse_note) if "note" in df.columns else pd.Series([{}] * len(df))
    meta_df = pd.json_normalize(meta)
    df = pd.concat([df, meta_df], axis=1)
    return df


def compute_R(df):
    if "realized_pnl" not in df.columns:
        raise ValueError("Нет realized_pnl")

    df = df.copy()
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    df["risk_usdt"] = pd.to_numeric(df["risk_usdt"], errors="coerce")
    df = df.dropna(subset=["realized_pnl", "risk_usdt"])
    df = df[df["risk_usdt"] > 0]
    df["R"] = df["realized_pnl"] / df["risk_usdt"]
    return df


def expectancy_by_score(df, bins=10):
    df = df.copy()
    df["edge_score"] = pd.to_numeric(df["edge_score"], errors="coerce")
    df = df.dropna(subset=["edge_score", "R"])
    df["score_bucket"] = pd.cut(df["edge_score"], bins=bins)
    grouped = df.groupby("score_bucket", observed=False)
    stats = grouped["R"].agg(["count", "mean", "std"])
    stats["winrate"] = grouped["R"].apply(lambda x: (x > 0).mean())
    return stats.sort_index()


def feature_expectancy(df, feature):
    df = df.copy()
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna(subset=[feature, "R"])
    df["bucket"] = pd.qcut(df[feature], q=5, duplicates="drop")
    grouped = df.groupby("bucket", observed=False)
    return grouped["R"].mean()


def feature_correlation(df):
    features = FEATURE_COLUMNS + ["edge_score"]
    corr_df = df.copy()
    for column in features + ["R"]:
        corr_df[column] = pd.to_numeric(corr_df[column], errors="coerce")
    corr_df = corr_df.dropna(subset=features + ["R"])
    return corr_df[features + ["R"]].corr()["R"].sort_values(ascending=False)


def fit_weights(df):
    model_df = df.copy()
    for column in FEATURE_COLUMNS + ["R"]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")
    model_df = model_df.dropna(subset=FEATURE_COLUMNS + ["R"])
    X = model_df[FEATURE_COLUMNS]
    y = model_df["R"]
    model = LinearRegression()
    model.fit(X, y)
    weights = dict(zip(X.columns, model.coef_))
    return weights, model.score(X, y)


def find_best_threshold(df):
    threshold_df = df.copy()
    threshold_df["edge_score"] = pd.to_numeric(threshold_df["edge_score"], errors="coerce")
    threshold_df = threshold_df.dropna(subset=["edge_score", "R"])

    thresholds = [i / 100 for i in range(20, 90, 5)]
    results = []
    for threshold in thresholds:
        subset = threshold_df[threshold_df["edge_score"] >= threshold]
        if len(subset) < 30:
            continue
        expectancy = subset["R"].mean()
        results.append((threshold, expectancy, len(subset)))
    return sorted(results, key=lambda x: x[1], reverse=True)


def plot_distribution(df, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(df["R"], bins=50)
    plt.title("PnL Distribution (R)")
    plt.xlabel("R")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="journal.csv")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--plot", default="r_distribution.png")
    args = parser.parse_args()

    df = load_journal(args.journal)
    df = compute_R(df)

    print("Expectancy by edge_score:")
    print(expectancy_by_score(df, bins=args.bins))
    print()

    print("Feature expectancy:")
    for feature in FEATURE_COLUMNS:
        print(feature)
        print(feature_expectancy(df, feature))
        print()

    print("Feature correlation:")
    print(feature_correlation(df))
    print()

    weights, r2 = fit_weights(df)
    print("Fitted weights:")
    print(weights)
    print(f"R^2: {r2:.4f}")
    print()

    print("Best thresholds:")
    for threshold, expectancy, count in find_best_threshold(df):
        print(f"threshold={threshold:.2f} expectancy={expectancy:.4f} count={count}")

    plot_distribution(df, Path(args.plot))


if __name__ == "__main__":
    main()
