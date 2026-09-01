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
    litros = None
    costo = None
    grafica_gas = None

    if request.method == "POST":
        tipo = request.form.get("tipo")

        if tipo == "notas":
            hours = float(request.form["hours"])
            result = LinearRegression.calculateGrade(hours)
            grafica_notas = LinearRegression.generarGraficaNotas(hours)

        elif tipo == "gasolina":
            km = float(request.form["km"])
            litros, costo = LinearRegression.calculateGasoline(km)
            grafica_gas = LinearRegression.generarGraficaGasolina(km)

    return render_template(
        "tempLinearRegression.html",
        result=result,
        grafica_notas=grafica_notas,
        litros=litros,
        costo=costo,
        grafica_gas=grafica_gas
    )

if __name__ == '__main__':
    app.run(debug=True)