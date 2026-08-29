from flask import Flask, render_template, request
import LinearRegression

app = Flask(__name__)

@app.route("/")
def template():
    return render_template("index.html")

@app.route("/LinearRegression", methods=["GET", "POST"])
def calculate():
    result = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        result = LinearRegression.calculateGrade(hours)
    return render_template("tempLinearRegression.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)