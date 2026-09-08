import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

CLASS_NAMES = {0: "Loss / Draw", 1: "Win"}

match_df = pd.read_csv("data/logistic_regression_match.csv")

X = match_df[["possession_pct"]]
y = match_df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

y_pred_test = logreg_model.predict(X_test)


def get_dataset_summary():
    return {
        "n_records": len(match_df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "independent_variable": "possession_pct (Possession percentage %)",
        "target_variable": "result (Match Result)",
        "class_0": "0 = Loss / Draw",
        "class_1": "1 = Win",
    }


def predict_result(possession_pct):
    row = pd.DataFrame([[possession_pct]], columns=["possession_pct"])
    prediction = logreg_model.predict(row)[0]
    probability = logreg_model.predict_proba(row)[0]
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

    plt.scatter(wins["possession_pct"], wins["result"], color="#2f9e6f",
                alpha=0.6, s=25, label="Win (1)")
    plt.scatter(losses["possession_pct"], losses["result"], color="#e0a872",
                alpha=0.6, s=25, label="Loss / Draw (0)")

    plt.title("Match Result vs Possession Percentage")
    plt.xlabel("Possession Percentage (%)")
    plt.ylabel("Match Result (0 = Loss/Draw, 1 = Win)")
    plt.yticks([0, 1])
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')


def generate_prediction_chart(possession_pct, prediction):
    plt.figure(figsize=(6.5, 4.2))
    wins = match_df[match_df["result"] == 1]
    losses = match_df[match_df["result"] == 0]

    plt.scatter(wins["possession_pct"], wins["result"], color="#2f9e6f",
                alpha=0.5, s=25, label="Win (1)")
    plt.scatter(losses["possession_pct"], losses["result"], color="#e0a872",
                alpha=0.5, s=25, label="Loss / Draw (0)")
    plt.scatter([possession_pct], [prediction], color="#D4A574", s=160,
                edgecolor="black", zorder=5, marker="D", label="New prediction")

    plt.title("New Classification on Training Data")
    plt.xlabel("Possession Percentage (%)")
    plt.ylabel("Match Result (0 = Loss/Draw, 1 = Win)")
    plt.yticks([0, 1])
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
