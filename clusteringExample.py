import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

FEATURES = ["poss", "sot", "gf", "ga", "home_advantage"]
TIER_NAMES = ["Low", "Moderate", "High"]
TIER_COLORS = {"Low": "#e0a872", "Moderate": "#D4A574", "High": "#2f9e6f"}

raw_df = pd.read_csv("data/matches.csv")
raw_df = raw_df.dropna(subset=["poss", "sot", "gf", "ga", "venue", "result"])
raw_df = raw_df.drop_duplicates()

match_df = raw_df.copy()
match_df["home_advantage"] = (match_df["venue"] == "Home").astype(int)
match_df["win"] = (match_df["result"] == "W").astype(int)

X = match_df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_labels = kmeans_model.fit_predict(X_scaled)
match_df["raw_cluster"] = raw_labels

silhouette = silhouette_score(X_scaled, raw_labels)

win_rate_by_cluster = match_df.groupby("raw_cluster")["win"].mean().sort_values()
ordered_raw_ids = win_rate_by_cluster.index.tolist()
label_map = {raw_id: TIER_NAMES[i] for i, raw_id in enumerate(ordered_raw_ids)}
match_df["cluster"] = match_df["raw_cluster"].map(label_map)

centroids_original = scaler.inverse_transform(kmeans_model.cluster_centers_)

CLUSTER_INTERPRETATION = {
    "Low": "Lower possession, fewer shots on target and a negative goal balance (goals scored below goals conceded) — the lowest observed win rate.",
    "Moderate": "Balanced possession and shot volume, with goals scored close to goals conceded — a middle win rate.",
    "High": "Higher possession, more shots on target and a positive goal balance — the highest observed win rate.",
}


def get_dataset_summary():
    return {
        "n_records": len(match_df),
        "features": [
            "poss (Possession percentage %)",
            "sot (Shots on target)",
            "gf (Average goals scored)",
            "ga (Average goals conceded)",
            "home_advantage (1 = Home, 0 = Away)",
        ],
        "source": "Kaggle - Football Match Statistics (team-match records)",
        "n_clusters": 3,
    }


def get_cluster_summary():
    summary = []
    for raw_id in ordered_raw_ids:
        name = label_map[raw_id]
        centroid = centroids_original[raw_id]
        cluster_rows = match_df[match_df["raw_cluster"] == raw_id]
        summary.append({
            "cluster": name,
            "n_records": int(len(cluster_rows)),
            "win_rate": round(float(cluster_rows["win"].mean()) * 100, 2),
            "centroid": {
                "poss": round(float(centroid[0]), 2),
                "sot": round(float(centroid[1]), 2),
                "gf": round(float(centroid[2]), 2),
                "ga": round(float(centroid[3]), 2),
                "home_advantage": round(float(centroid[4]), 2),
            },
        })
    return summary


def get_sample_records(n=25):
    columns = ["team", "opponent", "season", "poss", "sot", "gf", "ga", "home_advantage", "cluster"]
    return match_df[columns].head(n).to_dict(orient="records")


def get_silhouette_score():
    return round(float(silhouette), 4)


def generate_cluster_chart():
    plt.figure(figsize=(7, 4.6))
    for raw_id in ordered_raw_ids:
        name = label_map[raw_id]
        subset = match_df[match_df["raw_cluster"] == raw_id]
        plt.scatter(subset["poss"], subset["sot"], alpha=0.35, s=18,
                    color=TIER_COLORS[name], label=f"{name} win probability")

    plt.scatter(centroids_original[:, 0], centroids_original[:, 1],
                color="black", marker="X", s=160, zorder=5, label="Centroids")

    plt.title("K-Means Clusters: Possession vs Shots on Target")
    plt.xlabel("Possession Percentage (%)")
    plt.ylabel("Shots on Target")
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')


def get_cluster_interpretation():
    return CLUSTER_INTERPRETATION
