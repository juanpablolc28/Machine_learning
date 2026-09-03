from flask import Flask, render_template, request
import LinearRegression

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

@app.route("/use-cases")
def use_cases():
    return render_template("use_cases.html")

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

if __name__ == '__main__':
    app.run(debug=True)