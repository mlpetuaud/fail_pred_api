#https://ledatascientist.com/deployer-rapidement-des-modeles-de-ml-avec-fastapi/

#import sys
#sys.path.append('/home/asabuzz/python_ml_dl/api-dockers/')
from pydantic import BaseModel, Field
from fastapi import FastAPI
from joblib import load
import pandas as pd
from fastapi.responses import FileResponse
import os

app = FastAPI(title='Is this company likely to fail ?')

# On définit la forme de notre input
class InputFormulaire(BaseModel):
    dettes: int = Field(..., example=3000)
    statut: str = Field(..., example='Société par actions simplifiée')
    APE: str = Field(..., example='C')

#class Sentiment(BaseModel):
#    sentiment: str
#    sentiment_probability: float
# On définit les fonctions hors API qui permettent
# de récupérer le modèle
def get_model():
    #path = os.path.dirname(os.path.realpath(__file__))
    model = load('fit_model.joblib')
    return model


@app.get('/')
def get_root():
    return {'message': 'Welcome to the failure detection API'}
 
#@app.get('/predict')
#async def predict(message: str):
   #prediction_result = nlp(message)[0]
   #return Sentiment(sentiment=prediction_result['label'], sentiment_probability=float(prediction_result['score']))

@app.post("/predict")
async def predict(payload:InputFormulaire):
    # convert the payload to pandas DataFrame
    input_df = pd.DataFrame(payload.dict(), index=range(1))
    prediction_class = get_model().predict(input_df)[0]
    prediction_convert = {1:'fail', 0:'wont fail'}
    prediction = prediction_convert[prediction_class]
    predict_proba = get_model().predict_proba(input_df)[0][1]
    return {
        'prediction': prediction_class,
        'pred': prediction,
        'failure_proba': predict_proba
    }

#@app.get('/predict/{message}',response_model=Sentiment)
#async def predict(message: str):
#   prediction_result = nlp(message)[0]
#   return Sentiment(sentiment=prediction_result['label'], sentiment_probability=float(prediction_result['score']))


 



 
#@app.get('/predict/{message}',response_model=Sentiment)
#async def predict(message: str):
#   prediction_result = nlp(message)[0]
#   return Sentiment(sentiment=prediction_result['label'], sentiment_probability=float(prediction_result['score']))



#app = FastAPI()
#@app.get("/")
#async def root():
#    return {"message": "Hello World"}