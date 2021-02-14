# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 01:02:58 2021

@author: 91983
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle 
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in=open('model__KNN.pkl','rb')   
classifier=pickle.load(pickle_in)

@app.route('/',methods=['GET'])
def welcome():
    return "Welcome"

@app.route('/predict')
def prediction_auth():
    
    
    """ Authentication of the Banks Note(Single input) 
    The output of this Prediction will tell you whether the note is fake or not
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output value
        
    """
    
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,curtosis]])
    if str(prediction)=='1':
        return 'Genuine'
    else:
        return 'Forged'


@app.route('/predict_file',methods=['POST'])
def prediction_auth_file():
    
    """ Authentication of  the Banks Notes(file input) 
    The output of this Prediction will tell you whether the note is fake or not
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    
    file_test=pd.read_csv(request.files.get('file'))
    prediction=classifier.predict(file_test)
    prediction=str(np.where(prediction==1,'Genuine','Forged'))
    return prediction


if __name__=='__main__':
    app.run(host="0.0.0.0")