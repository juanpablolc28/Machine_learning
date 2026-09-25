import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "kmeans_manual_data.csv")
PLOTS_DIR = os.path.join("static", "plots", "manual_kmeans")
N_REGISTROS = 100
N_CLUSTERS = 3
N_ITERACIONES = 3
RANDOM_SEED = 42


def generar_datos(n=N_REGISTROS, seed=RANDOM_SEED, path=DATA_PATH):
    if os.path.exists(path):
        return pd.read_csv(path)

    rng = np.random.default_rng(seed)
    n_por_grupo = n // 3
    restante = n - n_por_grupo * 3

    grupo_dominante = pd.DataFrame({
        "possession": rng.normal(65, 6, n_por_grupo),
        "shots_on_target": rng.normal(11, 2, n_por_grupo),
    })
    grupo_equilibrado = pd.DataFrame({
        "possession": rng.normal(50, 5, n_por_grupo),
        "shots_on_target": rng.normal(6, 1.5, n_por_grupo),
    })
    grupo_defensivo = pd.DataFrame({
        "possession": rng.normal(35, 6, n_por_grupo + restante),
        "shots_on_target": rng.normal(3, 1.3, n_por_grupo + restante),
    })

    df = pd.concat([grupo_dominante, grupo_equilibrado, grupo_defensivo],
                    ignore_index=True)

    df["possession"] = df["possession"].clip(10, 90).round(1)
    df["shots_on_target"] = df["shots_on_target"].clip(0, 20).round().astype(int)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def centroides_iniciales(df):
    return np.array([
        [65.0, 11.0],
        [50.0, 6.0],
        [35.0, 3.0],
    ])


def distancia_euclidiana(punto, centroide):
    return math.sqrt(
        (punto[0] - centroide[0]) ** 2 + (punto[1] - centroide[1]) ** 2
    )


def matriz_distancias(df, centroides):
    puntos = df[["possession", "shots_on_target"]].to_numpy()
    dist = np.zeros((len(puntos), len(centroides)))
    for i, punto in enumerate(puntos):
        for j, centroide in enumerate(centroides):
            dist[i, j] = distancia_euclidiana(punto, centroide)
    return dist


def asignar_y_recalcular(df, centroides):
    dist = matriz_distancias(df, centroides)
    df = df.copy()
    df["cluster"] = np.argmin(dist, axis=1)

    nuevos_centroides = centroides.copy()
    for k in range(len(centroides)):
        puntos_cluster = df[df["cluster"] == k][["possession", "shots_on_target"]]
        if len(puntos_cluster) > 0:
            nuevos_centroides[k] = puntos_cluster.mean().to_numpy()

    return df, nuevos_centroides, dist


def tabla_distancias(df, dist):
    df = df.reset_index(drop=True)
    filas = []
    for i, row in df.iterrows():
        fila = {
            "id": int(row["id"]),
            "possession": row["possession"],
            "shots_on_target": row["shots_on_target"],
            "cluster": int(row["cluster"]),
        }
        for k in range(dist.shape[1]):
            fila[f"dist_c{k}"] = round(float(dist[i, k]), 2)
        filas.append(fila)
    return filas


def varianza_por_cluster(df, centroides):
    resultados = {}
    for k in range(len(centroides)):
        puntos_cluster = df[df["cluster"] == k][["possession", "shots_on_target"]].to_numpy()
        if len(puntos_cluster) == 0:
            resultados[k] = 0.0
            continue
        sse = sum(distancia_euclidiana(p, centroides[k]) ** 2 for p in puntos_cluster)
        resultados[k] = round(sse, 2)
    return resultados


def graficar(df, centroides, titulo, nombre_archivo, con_cluster=True):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(6, 5))

    if con_cluster and "cluster" in df.columns:
        colores = ["#4C72B0", "#DD8452", "#55A868"]
        for k in range(len(centroides)):
            subset = df[df["cluster"] == k]
            plt.scatter(subset["possession"], subset["shots_on_target"],
                        color=colores[k % len(colores)], label=f"Cluster {k}",
                        alpha=0.7, s=35)
    else:
        plt.scatter(df["possession"], df["shots_on_target"],
                     color="gray", alpha=0.6, s=35, label="Datos")

    plt.scatter(centroides[:, 0], centroides[:, 1], color="black", marker="X",
                s=200, label="Centroides", edgecolors="white", linewidths=1.5)

    plt.title(titulo)
    plt.xlabel("Possession (%)")
    plt.ylabel("Shots on target")
    plt.legend()
    plt.tight_layout()

    ruta = os.path.join(PLOTS_DIR, nombre_archivo)
    plt.savefig(ruta, dpi=110)
    plt.close()
    return ruta


def ejecutar_kmeans_manual():
    df = generar_datos()
    centroides = centroides_iniciales(df)

    resultados_por_iteracion = []

    graficar(df, centroides, "Situacion inicial", "iteracion_0_inicial.png",
              con_cluster=False)
    for i in range(1, N_ITERACIONES + 1):
        df, centroides, dist = asignar_y_recalcular(df, centroides)
        sse = varianza_por_cluster(df, centroides)
        tabla = tabla_distancias(df, dist)
        ruta_plot = graficar(df, centroides, f"Iteracion {i}",
                              f"iteracion_{i}.png", con_cluster=True)

        resultados_por_iteracion.append({
            "iteracion": i,
            "centroides": centroides.copy(),
            "sse_por_cluster": sse,
            "sse_total": round(sum(sse.values()), 2),
            "plot": ruta_plot,
            "tabla": tabla,
        })

    return df, centroides, resultados_por_iteracion


if __name__ == "__main__":
    df_final, centroides_finales, historial = ejecutar_kmeans_manual()
    for r in historial:
        print(r["iteracion"], r["centroides"].tolist(), r["sse_por_cluster"], r["sse_total"])
