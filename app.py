from flask import Flask, render_template, request
import LinearRegression

app = Flask(__name__)

@app.route("/")
def template():
    return render_template("index.html")

# --- Rutas nuevas ---

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

# --- Fin de rutas nuevas ---

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

@app.route("/concepts")
def concepts():
    return render_template("concepts.html")

@app.route("/types")
def types():
    return render_template("types.html")

@app.route("/linear-regression-concepts")
def linear_regression_concepts():
    return render_template("linear_regression_concepts.html")

@app.route("/use-cases")
def use_cases():
    return render_template("use_cases.html")

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

if __name__ == '__main__':
    app.run(debug=True)