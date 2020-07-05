from flask import Flask,request, url_for, redirect, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

xgf=joblib.load(open('blackfriday.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("main.html")


@app.route('/predict',methods=['POST','GET'])
def predict():

    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    columns = ["Age", "Occupation", "City_Category","Product_Category_1","Product_Category_2","Product_Category_3"]
    pred_args=pd.DataFrame(final,columns=columns,dtype='float',index=['input'])
    prediction=xgf.predict(pred_args)


    return render_template('main.html',pred=prediction)

if __name__ == '__main__':
    app.run()
