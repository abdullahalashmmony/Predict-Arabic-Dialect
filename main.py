from copyreg import pickle
from pyexpat import model
from flask import Flask,render_template,request
from matplotlib.pyplot import clf
import pandas as pd
import pickle

app = Flask(__name__)


#model=pickle.load(open('model.pkl','rb'))
vect=pickle.load(open('vectorizer.pkl','rb'))
pipe=pickle.load(open('pipeline.pkl','rb'))


@app.route('/')

def home():
	return render_template('home.html')

@app.route('/result',methods=['POST','GET'])
def result():
        result_pred = pipe.predict(vect([result]))
        return render_template('result.html',result=result_pred)


if __name__ == '__main__':
	app.run(debug=True)