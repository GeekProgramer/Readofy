from classification import clf
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def main():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def process():
    input_value = request.form['text']
    val = clf(input_value)
    print(val)
    return render_template('result.html', val=val)
@app.route('/index', methods=['POST'])
def index():
    return render_template('index.html')


app.run(debug=True)
