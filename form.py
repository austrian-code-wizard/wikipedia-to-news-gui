from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired


class InputForm(FlaskForm):
    text = TextAreaField('Input Text', validators=[DataRequired()], render_kw={'class': 'form-control', 'rows': 12})
    submit = SubmitField('Analyze')
