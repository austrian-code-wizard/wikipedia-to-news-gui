from flask import Flask, render_template, flash, redirect
from flask_bootstrap import Bootstrap
from config import Config
from form import InputForm
from model import Analyzer

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config.from_object(Config)
app.config["analyzer"] = Analyzer(model="2020-03-15T23:57:24--acc--0.846.pt")

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = InputForm()
    if form.validate_on_submit():
        flash("Analyzing Text...")
        sentences, labels = app.config["analyzer"].analyze(form.text.data)
        return render_template('result.html', sentences=sentences, labels=labels)
    return render_template('index.html', title='Analyze', form=form)


if __name__ == '__main__':
    app.run()
