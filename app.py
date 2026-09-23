from flask import Flask, render_template, request
import LinearRegression
import LogisticRegressionModel
import LDAModel
import clusteringExample

app = Flask(__name__)

@app.route("/")
def template():
    return render_template("index.html")

@app.route("/concepts")
def concepts():
    return render_template("concepts.html")

@app.route("/types")
def types():
    return render_template("types.html")

@app.route("/use-case-1")
def use_case_1():
    return render_template("use_case_1.html")

@app.route("/use-case-2")
def use_case_2():
    return render_template("use_case_2.html")

@app.route("/use-case-3")
def use_case_3():
    return render_template("use_case_3.html")

@app.route("/use-case-4")
def use_case_4():
    return render_template("use_case_4.html")

@app.route("/linear-regression-concepts")
def linear_regression_concepts():
    return render_template("linear_regression_concepts.html")

@app.route("/LinearRegression", methods=["GET", "POST"])
def calculate():
    liters = None
    cost = None
    co2 = None
    gas_chart = None

    if request.method == "POST":
        tipo = request.form.get("tipo")

        if tipo == "gasolina":
            km = float(request.form["km"])
            liters, cost, co2 = LinearRegression.calculateGasoline(km)
            gas_chart = LinearRegression.generateGasolineChart(km)

    return render_template(
        "tempLinearRegression.html",
        liters=liters,
        cost=cost,
        co2=co2,
        gas_chart=gas_chart
    )

@app.route("/logistic-regression-concepts")
def logistic_regression_concepts():
    return render_template("logistic_regression_concepts.html")


@app.route("/LogisticRegression", methods=["GET", "POST"])
def logistic_regression_application():
    result = None
    error = None

    if request.method == "POST":
        try:
            possession_pct = float(request.form["possession_pct"])
            if not (0 <= possession_pct <= 100):
                raise ValueError("Possession percentage must be between 0 and 100.")
            result = LogisticRegressionModel.predict_result(possession_pct)
            result["chart"] = LogisticRegressionModel.generate_prediction_chart(
                possession_pct, result["prediction"]
            )
            result["possession_pct"] = possession_pct
        except (ValueError, KeyError):
            error = "Please enter a valid possession percentage between 0 and 100."

    return render_template(
        "logistic_regression_application.html",
        summary=LogisticRegressionModel.get_dataset_summary(),
        dataset_chart=LogisticRegressionModel.generate_dataset_chart(),
        result=result,
        error=error,
    )


@app.route("/logistic-regression-metrics")
def logistic_regression_metrics():
    return render_template(
        "logistic_regression_metrics.html",
        metrics=LogisticRegressionModel.get_evaluation_metrics(),
    )


@app.route("/lda-concepts")
def lda_concepts():
    return render_template("lda_concepts.html")


@app.route("/LDA", methods=["GET", "POST"])
def lda_application():
    result = None
    error = None

    if request.method == "POST":
        try:
            possession_pct = float(request.form["possession_pct"])
            shots_on_target = float(request.form["shots_on_target"])
            avg_goals_scored = float(request.form["avg_goals_scored"])
            home_advantage = int(request.form["home_advantage"])

            if not (0 <= possession_pct <= 100):
                raise ValueError("Invalid possession percentage.")
            if shots_on_target < 0 or avg_goals_scored < 0:
                raise ValueError("Values must be non-negative.")

            result = LDAModel.predict_result(
                possession_pct, shots_on_target, avg_goals_scored, home_advantage
            )
            result["chart"] = LDAModel.generate_prediction_chart(
                possession_pct, shots_on_target, result["prediction"]
            )
            result["possession_pct"] = possession_pct
            result["shots_on_target"] = shots_on_target
            result["avg_goals_scored"] = avg_goals_scored
            result["home_advantage"] = home_advantage
        except (ValueError, KeyError):
            error = "Please fill in all fields with valid numeric values."

    return render_template(
        "lda_application.html",
        summary=LDAModel.get_dataset_summary(),
        dataset_chart=LDAModel.generate_dataset_chart(),
        result=result,
        error=error,
    )


@app.route("/lda-metrics")
def lda_metrics():
    return render_template(
        "lda_metrics.html",
        metrics=LDAModel.get_evaluation_metrics(),
    )


@app.route("/kmeans-concepts")
def kmeans_concepts():
    return render_template("kmeans_concepts.html")


@app.route("/manual-exercise")
def manual_exercise():
    return render_template("manual_exercise.html")


@app.route("/clustering-application")
def clustering_application():
    return render_template(
        "clustering_application.html",
        summary=clusteringExample.get_dataset_summary(),
        cluster_summary=clusteringExample.get_cluster_summary(),
        sample_records=clusteringExample.get_sample_records(),
        silhouette=clusteringExample.get_silhouette_score(),
        chart=clusteringExample.generate_cluster_chart(),
        interpretation=clusteringExample.get_cluster_interpretation(),
    )


if __name__ == '__main__':
    app.run(debug=True)