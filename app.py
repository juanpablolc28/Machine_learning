from flask import Flask, render_template
from flask import Flask

app = Flask(__name__)
@app.route("/")
def home():
    return "Hello, World!"


@app.route("/template")
def template():
    return render_template("index.html")
