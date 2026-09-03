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
    result = None
    grafica_notas = None
    liters = None
    cost = None
    co2 = None
    gas_chart = None

    if request.method == "POST":
        tipo = request.form.get("tipo")

        if tipo == "notas":
            hours = float(request.form["hours"])
            result = LinearRegression.calculateGrade(hours)
            grafica_notas = LinearRegression.generarGraficaNotas(hours)

        elif tipo == "gasolina":
            km = float(request.form["km"])
            liters, cost, co2 = LinearRegression.calculateGasoline(km)
            gas_chart = LinearRegression.generateGasolineChart(km)

    return render_template(
        "tempLinearRegression.html",
        result=result,
        grafica_notas=grafica_notas,
        liters=liters,
        cost=cost,
        co2=co2,
        gas_chart=gas_chart
    )

if __name__ == '__main__':
    app.run(debug=True)