from flask import Flask, render_template, request
import LinearRegression

app = Flask(__name__)

@app.route("/")
def template():
    return render_template("index.html")

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