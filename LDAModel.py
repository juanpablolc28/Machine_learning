import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

CLASS_NAMES = {0: "Loss / Draw", 1: "Win"}
FEATURES = ["possession_pct", "shots_on_target", "avg_goals_scored", "home_advantage"]

match_df = pd.read_csv("data/lda_match.csv")

X = match_df[FEATURES]
y = match_df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

y_pred_test = lda_model.predict(X_test)


def get_dataset_summary():
    return {
        "n_records": len(match_df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "independent_variables": [
            "possession_pct (Possession percentage %)",
            "shots_on_target (Shots on target)",
            "avg_goals_scored (Average goals scored)",
            "home_advantage (1 = Home, 0 = Away)",
        ],
        "target_variable": "result (Match Result)",
        "class_0": "0 = Loss / Draw",
        "class_1": "1 = Win",
    }


def predict_result(possession_pct, shots_on_target, avg_goals_scored, home_advantage):
    row = pd.DataFrame(
        [[possession_pct, shots_on_target, avg_goals_scored, home_advantage]],
        columns=FEATURES,
    )
    prediction = lda_model.predict(row)[0]
    probability = lda_model.predict_proba(row)[0]
    return {
        "prediction": int(prediction),
        "class_name": CLASS_NAMES[int(prediction)],
        "prob_win": round(float(probability[1]) * 100, 2),
        "prob_loss": round(float(probability[0]) * 100, 2),
    }


def generate_dataset_chart():
    plt.figure(figsize=(6.5, 4.2))
    wins = match_df[match_df["result"] == 1]
    losses = match_df[match_df["result"] == 0]

    plt.scatter(wins["possession_pct"], wins["shots_on_target"], color="#1b7a4d",
                edgecolor="white", linewidth=0.4, alpha=0.85, s=32, label="Win (1)")
    plt.scatter(losses["possession_pct"], losses["shots_on_target"], color="#d9822b",
                edgecolor="white", linewidth=0.4, alpha=0.85, s=32, label="Loss / Draw (0)")
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    avg_goals_mean = match_df["avg_goals_scored"].mean()
    home_mean = match_df["home_advantage"].mean()

    w = lda_model.coef_[0]
    b = lda_model.intercept_[0]

    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(w[0] * x_vals + w[2] * avg_goals_mean + w[3] * home_mean + b) / w[1]

    plt.plot(x_vals, y_vals, color="#333333", linewidth=1.8, label="Decision boundary")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title("Match Result: Possession vs Shots on Target")
    plt.xlabel("Possession Percentage (%)")
    plt.ylabel("Shots on Target")
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')


def generate_prediction_chart(possession_pct, shots_on_target, prediction):
    plt.figure(figsize=(6.5, 4.2))
    wins = match_df[match_df["result"] == 1]
    losses = match_df[match_df["result"] == 0]

    plt.scatter(wins["possession_pct"], wins["shots_on_target"], color="#1b7a4d",
                edgecolor="white", linewidth=0.4, alpha=0.85, s=32, label="Win (1)")
    plt.scatter(losses["possession_pct"], losses["shots_on_target"], color="#d9822b",
                edgecolor="white", linewidth=0.4, alpha=0.85, s=32, label="Loss / Draw (0)")
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    avg_goals_mean = match_df["avg_goals_scored"].mean()
    home_mean = match_df["home_advantage"].mean()

    w = lda_model.coef_[0]
    b = lda_model.intercept_[0]

    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(w[0] * x_vals + w[2] * avg_goals_mean + w[3] * home_mean + b) / w[1]

    plt.plot(x_vals, y_vals, color="#333333", linewidth=1.8, label="Decision boundary")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.scatter([possession_pct], [shots_on_target], color="#D4A574", s=160,
                edgecolor="black", zorder=5, marker="D", label="New prediction")

    plt.title("New Classification on Training Data")
    plt.xlabel("Possession Percentage (%)")
    plt.ylabel("Shots on Target")
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')


def get_evaluation_metrics():
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "n_test": len(y_test),
    }
