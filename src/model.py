import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

SKIP_COLS = ["filename", "label", "label_name", "source_dir"]


def load_data(csv_path: str):
    df = pd.read_csv(csv_path).dropna()
    feature_cols = [c for c in df.columns if c not in SKIP_COLS]
    X = df[feature_cols].values
    y = df["label"].values
    print(f"  Samples  : {len(df)} ({sum(y==0)} natural, {sum(y==1)} AI)")
    print(f"  Features : {len(feature_cols)}")
    return X, y, feature_cols, df


def split_and_save(df, train_ratio=0.70, dev_ratio=0.15):
    """Stratified 3-way split → saves train/dev/eval CSVs, returns indices."""
    eval_ratio = round(1.0 - train_ratio - dev_ratio, 4)

    df_train, df_temp = train_test_split(
        df, test_size=(dev_ratio + eval_ratio), stratify=df["label"], random_state=42
    )
    dev_share = dev_ratio / (dev_ratio + eval_ratio)
    df_dev, df_eval = train_test_split(
        df_temp, test_size=(1 - dev_share), stratify=df_temp["label"], random_state=42
    )

    os.makedirs("results", exist_ok=True)
    df_train.to_csv("results/train.csv", index=False)
    df_dev.to_csv("results/dev.csv",     index=False)
    df_eval.to_csv("results/eval.csv",   index=False)

    print(f"\n  {'Split':<10} {'Total':>7} {'Natural':>9} {'AI':>9}")
    print(f"  {'-'*38}")
    for name, d in [("Train", df_train), ("Dev", df_dev), ("Eval", df_eval)]:
        print(f"  {name:<10} {len(d):>7} {sum(d['label']==0):>9} {sum(d['label']==1):>9}")

    print(f"\n  Saved → results/train.csv / dev.csv / eval.csv")
    return df_train, df_dev, df_eval


def train(csv_path: str = "results/dataset.csv", model_out: str = "models/detector.pkl"):
    print("\n" + "="*55)
    print("TRAINING AI AUDIO DETECTOR")
    print("="*55)

    X, y, feature_cols, df = load_data(csv_path)
    os.makedirs("models", exist_ok=True)

    # ── 3-way split: 70 / 15 / 15 ────────────────────────────────
    print("\n── Dataset Split (70 / 15 / 15) ───────────────────")
    df_train, df_dev, df_eval = split_and_save(df)

    feature_cols = [c for c in df.columns if c not in SKIP_COLS]
    X_train = df_train[feature_cols].values;  y_train = df_train["label"].values
    X_dev   = df_dev[feature_cols].values;    y_dev   = df_dev["label"].values
    X_eval  = df_eval[feature_cols].values;   y_eval  = df_eval["label"].values

    # ── Models to compare ─────────────────────────────────────────
    models = {
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42
            ))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ))
        ]),
    }

    # ── Cross-validation on train split ───────────────────────────
    print("\n── Cross-Validation on Train split (5-fold) ───────")
    best_score, best_name, best_model = 0, None, None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"  {name:<20} AUC: {scores.mean():.3f} ± {scores.std():.3f}")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name  = name
            best_model = model

    print(f"\n  Best model: {best_name} (CV AUC: {best_score:.3f})")

    # ── Train on train split ───────────────────────────────────────
    best_model.fit(X_train, y_train)

    # ── Dev evaluation ────────────────────────────────────────────
    print("\n── Dev Set Results ────────────────────────────────")
    y_dev_pred = best_model.predict(X_dev)
    y_dev_prob = best_model.predict_proba(X_dev)[:, 1]
    print(classification_report(y_dev, y_dev_pred, target_names=["Natural", "AI"]))
    print(f"  ROC-AUC : {roc_auc_score(y_dev, y_dev_prob):.4f}")
    plot_confusion_matrix(y_dev, y_dev_pred, save_path="results/confusion_matrix_dev.png")

    # ── Final eval (held-out) ─────────────────────────────────────
    print("\n── Eval Set Results (held-out) ────────────────────")
    y_eval_pred = best_model.predict(X_eval)
    y_eval_prob = best_model.predict_proba(X_eval)[:, 1]
    print(classification_report(y_eval, y_eval_pred, target_names=["Natural", "AI"]))
    print(f"  ROC-AUC : {roc_auc_score(y_eval, y_eval_prob):.4f}")
    plot_confusion_matrix(y_eval, y_eval_pred, save_path="results/confusion_matrix_eval.png")

    # ── Feature importance ────────────────────────────────────────
    plot_feature_importance(best_model, feature_cols, best_name)

    # ── Save model ────────────────────────────────────────────────
    with open(model_out, "wb") as f:
        pickle.dump((best_model, feature_cols), f)

    print(f"\n  ✅ Model saved to {model_out}")
    print("="*55)
    return best_model, feature_cols


def plot_confusion_matrix(y_test, y_pred, save_path="results/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#12121a")
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma",
                xticklabels=["Natural", "AI"],
                yticklabels=["Natural", "AI"], ax=ax)
    ax.set_title("Confusion Matrix", color="#e2e8f0")
    ax.set_ylabel("True Label", color="#e2e8f0")
    ax.set_xlabel("Predicted Label", color="#e2e8f0")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✅ Saved: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_cols, model_name, save_path="results/feature_importance.png"):
    try:
        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # top 20

            fig, ax = plt.subplots(figsize=(12, 7))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#12121a")

            colors = ["#00ffc8" if i < 5 else "#ff6b35" if i < 10 else "#64748b" for i in range(len(indices))]
            ax.barh(
                [feature_cols[i] for i in indices][::-1],
                importances[indices][::-1],
                color=colors[::-1]
            )
            ax.set_title(f"Top 20 Feature Importances ({model_name})\nGreen = strongest AI detectors",
                        color="#e2e8f0", fontsize=12)
            ax.set_xlabel("Importance Score", color="#e2e8f0")
            ax.tick_params(colors="#e2e8f0")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✅ Saved: {save_path}")
            plt.close()
    except Exception as e:
        print(f"  Feature importance plot skipped: {e}")


if __name__ == "__main__":
    train()