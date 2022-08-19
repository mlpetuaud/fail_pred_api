#https://ledatascientist.com/deployer-rapidement-des-modeles-de-ml-avec-fastapi/

#import sys
#sys.path.append('/home/asabuzz/python_ml_dl/api-dockers/')
from pydantic import BaseModel, Field
from fastapi import FastAPI
from joblib import load
import pandas as pd
from fastapi.responses import FileResponse
import os
from prepare_dataset import prepare_dataset_api

"""
    This file creates and handles the FastAPI app giving access to 
    a company failure prediction application
"""

# Create FastAPI app
app = FastAPI(title='Is this company likely to fail ?')

# Define a class inheriting from BaseModel to handle and validate
# the user input
class InputFormulaire(BaseModel):
    A: str = Field(..., example='4511Z')
    B: str = Field(..., example='05/04/1985')
    C: int = Field(..., example=0)
    D: int = Field(..., example=2897808)
    E: int = Field(..., example=0)
    F: int = Field(..., example=0)
    G: int = Field(..., example=68462)
    H: int = Field(..., example=0)
    I: int = Field(..., example=20562168)
    J: int = Field(..., example=4440)
    BA: int = Field(..., example=6225974)
    BB: int = Field(..., example=4096665)
    BC: int = Field(..., example=0)
    BD: int = Field(..., example=23)
    BE: int = Field(..., example=850613)
    BF: int = Field(..., example=54661)
    BG: int = Field(..., example=31863007)
    BH: int = Field(..., example=0)
    BI: int = Field(..., example=0)
    BJ: int = Field(..., example=34760815)
    CA: int = Field(..., example=2930800)
    CB: int = Field(..., example=1492439)
    CC: int = Field(..., example=715086)
    CD: int = Field(..., example=5869950)
    CE: int = Field(..., example=0)
    CF: int = Field(..., example=207524)
    CG: int = Field(..., example=0)
    CH: int = Field(..., example=0)
    CI: int = Field(..., example=9831341)
    CJ: int = Field(..., example=15000)
    DA: int = Field(..., example=61331)
    DB: int = Field(..., example=16666544)
    DC: int = Field(..., example=1765117)
    DD: int = Field(..., example=0)
    DE: int = Field(..., example=303337)
    DF: int = Field(..., example=0)
    DG: int = Field(..., example=40671)
    DH: int = Field(..., example=28683341)
    DI: int = Field(..., example=34760815)
    DJ: int = Field(..., example=0)
    EA: int = Field(..., example=95679937)
    EB: int = Field(..., example=0)
    EC: int = Field(..., example=95679937)
    ED: int = Field(..., example=7017)
    EE: int = Field(..., example=696822)
    EF: int = Field(..., example=96433879)
    EG: int = Field(..., example=83900620)
    EH: int = Field(..., example=-1448891)
    EI: int = Field(..., example=0)
    EJ: int = Field(..., example=0)
    FA: int = Field(..., example=5958909)
    FB: int = Field(..., example=541581)
    FC: int = Field(..., example=3950591)
    FD: int = Field(..., example=1714123)
    FE: int = Field(..., example=369187)
    FF: int = Field(..., example=44100)
    FG: int = Field(..., example=188979)
    FH: int = Field(..., example=7640)
    FI: int = Field(..., example=5910)
    FJ: int = Field(..., example=95232748)
    GA: int = Field(..., example=1201132)
    GB: int = Field(..., example=85730)
    GC: int = Field(..., example=310602)
    GD: int = Field(..., example=310602)
    GE: int = Field(..., example=0)
    GF: int = Field(..., example=2700)
    GG: int = Field(..., example=12225)
    GH: int = Field(..., example=0)
    GI: int = Field(..., example=0)
    GJ: int = Field(..., example=5732)
    HA: int = Field(..., example=6493)
    HB: int = Field(..., example=715086)
    HC: int = Field(..., example=66765)
    HD: int = Field(..., example=200902)
    HE: int = Field(..., example=186195)
    HF: int = Field(..., example=620560)
    HG: int = Field(..., example=0)
    HH: int = Field(..., example=392807)
    HI: int = Field(..., example=0)
    HJ: int = Field(..., example=0)
    IA: int = Field(..., example=0)
    IB: int = Field(..., example=0)
    IC: int = Field(..., example=47500)
    ID: int = Field(..., example=0)
    IE: int = Field(..., example=0)
    IF: int = Field(..., example=0)
    IG: int = Field(..., example=976260)
    IH: int = Field(..., example=12)
    II: str = Field(..., example='Société par actions simplifiée')


def prepare_input(payload):
    """his function transforms the user input into a DataFrame
        with all information needed to perform failure prediction

    Args:
        payload (InputFormulaire): user's input

    Returns:
        pandas DataFrame: dataframe holding the user input into
        the required format to be used by the ML pipeline
    """
    decoding_dict = {'A': 'Code APE',
                    'B': 'Date de création',
                    'C': 'Capital souscrit non appelé (I) (AA) 2018 (€)',
                    'D': 'TOTAL (II) (net) (BJNET) 2018 (€)',
                    'E': 'Matières premières, approvisionnements (net) (BLNET) 2018 (€)',
                    'F': 'En cours de production de biens (net) (BNNET) 2018 (€)',
                    'G': 'En cours de production de services (net) (BPNET) 2018 (€)',
                    'H': 'Produits intermédiaires et finis (net) (BRNET) 2018 (€)',
                    'I': 'Marchandises (net) (BTNET) 2018 (€)',
                    'J': 'Avances et acomptes versés sur commandes (net) (BVNET) 2018 (€)',
                    'BA': 'Clients et comptes rattachés (3) (net) (BXNET) 2018 (€)',
                    'BB': 'Autres créances (3) (net) (BZNET) 2018 (€)',
                    'BC': 'Capital souscrit et appelé, non versé (net) (CBNET) 2018 (€)',
                    'BD': 'Valeurs mobilières de placement (net) (CDNET) 2018 (€)',
                    'BE': 'Disponibilités (net) (CFNET) 2018 (€)',
                    'BF': "Charges constatées d'avance (3) (net) (CHNET) 2018 (€)",
                    'BG': 'TOTAL (III) (net) (CJNET) 2018 (€)',
                    'BH': 'Primes de remboursement des obligations (CM) 2018 (€)',
                    'BI': 'Ecarts de conversion actif (CN) 2018 (€)',
                    'BJ': 'TOTAL GENERAL(I à VI) (net) (CONET) 2018 (€)',
                    'CA': 'Capital social ou individuel (1) (DA) 2018 (€)',
                    'CB': 'Report à nouveau (DH) 2018 (€)',
                    'CC': "RESULTAT DE L'EXERCICE (bénéfice ou perte) (DI) 2018 (€)",
                    'CD': 'TOTAL (I) (DL) 2018 (€)',
                    'CE': 'TOTAL(II) (DO) 2018 (€)',
                    'CF': 'TOTAL (III) (DR) 2018 (€)',
                    'CG': 'Autres emprunts obligataires (DT) 2018 (€)',
                    'CH': 'Emprunts obligataires convertibles (DS) 2018 (€)',
                    'CI': 'Emprunts et dettes auprès des établissements de crédit (5) (DU) 2018 (€)',
                    'CJ': 'Emprunts et dettes financières divers (DV) 2018 (€)',
                    'DA': 'Avances et acomptes reçus sur commandes en cours (DW) 2018 (€)',
                    'DB': 'Dettes fournisseurs et comptes rattachés (DX) 2018 (€)',
                    'DC': 'Dettes fiscales et sociales (DY) 2018 (€)',
                    'DD': 'Dettes sur immobilisations et comptes rattachés (DZ) 2018 (€)',
                    'DE': 'Autres dettes (EA) 2018 (€)',
                    'DF': "dont comptes courants d'associés de l'exercice N (EA2) 2018 (€)",
                    'DG': "Produits constatés d'avance (EB) 2018 (€)",
                    'DH': 'TOTAL (IV) (EC) 2018 (€)',
                    'DI': 'TOTAL GENERAL (I à V) (EE) 2018 (€)',
                    'DJ': '(5)\xa0Dont concours bancaires courants, et soldes créditeurs de banques et CCP (EH) 2018 (€)',
                    'EA': "Chiffre d'affaires net (France) (FJ) 2018 (€)",
                    'EB': "Chiffre d'affaires net (Exportations et livraisons intracommunautaires) (FK) 2018 (€)",
                    'EC': "Chiffre d'affaires net (Total) (FL) 2018 (€)",
                    'ED': "Subventions d'exploitation (FO) 2018 (€)",
                    'EE': 'Reprises sur amortissements et provisions, transferts de charges (9) (FP) 2018 (€)',
                    'EF': "Total des produits d'exploitation (2) (I) (FR) 2018 (€)",
                    'EG': 'Achats de marchandises (y compris droits de douane) (FS) 2018 (€)',
                    'EH': 'Variation de stock (marchandises) (FT) 2018 (€)',
                    'EI': 'Achats de matières premières et autres approvisionnements (y compris droits de douane) (FU) 2018 (€)',
                    'EJ': 'Variation de stock (matières premières et approvisionnements) (FV) 2018 (€)',
                    'FA': 'Autres achats et charges externes (3) (6 bis) (FW) 2018 (€)',
                    'FB': 'Impôts, taxes et versements assimilés (FX) 2018 (€)',
                    'FC': 'Salaires et traitements (FY) 2018 (€)',
                    'FD': 'Charges sociales (10) (FZ) 2018 (€)',
                    'FE': "Dotations d'exploitation sur immobilisations (dotations aux amortissements) (GA) 2018 (€)",
                    'FF': "Dotations d'exploitation sur immobilisations (dotations aux provisions) (GB) 2018 (€)",
                    'FG': "Dotations d'exploitation sur actif circulant (dotations aux provisions) (GC) 2018 (€)",
                    'FH': "Dotations d'exploitation pour risques et charges (dotations aux provisions) (GD) 2018 (€)",
                    'FI': 'Autres charges (12) (GE) 2018 (€)',
                    'FJ': "Total des charges d'exploitation (4) (II) (GF) 2018 (€)",
                    'GA': "1 - RESULTAT D'EXPLOITATION (I - II) (GG) 2018 (€)",
                    'GB': 'Total des produits financiers (V) (GP) 2018 (€)',
                    'GC': 'Intérêts et charges assimilées (GR) 2018 (€)',
                    'GD': 'Total des charges financières (VI) (GU) 2018 (€)',
                    'GE': 'Dotations financières aux amortissements et provisions (GQ) 2018 (€)',
                    'GF': 'Reprises sur provisions & transferts de charges (GM) 2018 (€)',
                    'GG': 'Total des produits exceptionnels (VII) (HD) 2018 (€)',
                    'GH': 'Reprises sur provisions & transferts de charges (HC) 2018 (€)',
                    'GI': 'Dotations exceptionnelles aux amortissements et provisions (6 ter) (HG) 2018 (€)',
                    'GJ': 'Total des charges exceptionnelles (VIII) (HH) 2018 (€)',
                    'HA': '4 - RESULTAT EXCEPTIONNEL (VII - VIII) (HI) 2018 (€)',
                    'HB': '5 - BENEFICE OU PERTE (Total des produits - total des charges) (HN) 2018 (€)',
                    'HC': "Participation des salariés aux résultats de l'entreprise (HJ) 2018 (€)",
                    'HD': 'Impôts sur les bénéfices (HK) 2018 (€)',
                    'HE': 'Clients douteux ou litigieux - Montant brut (VA) 2018 (€)',
                    'HF': 'Sécurité sociale et autres organismes sociaux - Montant brut (8D) 2018 (€)',
                    'HG': 'Impôts sur les bénéfices - Montant brut (8E) 2018 (€)',
                    'HH': 'T.V.A. - Montant brut (VW) 2018 (€)',
                    'HI': 'Emprunts souscrits en cours d’exercice - à 1 an au plus (VJ2) 2018 (€)',
                    'HJ': 'Effets portés à l’escompte et non échus (YS) 2018 (€)',
                    'IA': 'Sous‐traitance (YT) 2018 (€)',
                    'IB': 'Ecarts de conversion passif (V) (ED) 2018 (€)',
                    'IC': 'Production stockée (FM) 2018 (€)',
                    'ID': 'Production immobilisée (FN) 2018 (€)',
                    'IE': '(3)\xa0Dont Crédit-bail mobilier (HP) 2018 (€)',
                    'IF': '(3)\xa0Dont Crédit-bail immobilier (HQ) 2018 (€)',
                    'IG': '3 - RESULTAT COURANT AVANT IMPOTS (I - II + III - IV + V - VI) (GW) 2018 (€)',
                    'IH': "Nombre de mois de l'exercice comptable 2018",
                    'II': 'Catégorie juridique (Niveau II)'}
    input_dict = {decoding_dict[key]:value for (key, value) in payload.dict().items()}
    input_df = pd.DataFrame(input_dict, index=range(1))
    df = prepare_dataset_api(input_df)
    return df

def get_model():
    """This function loads a sklearn fit model stored as a joblib file
    and returns it
    Returns:
        model (sklearn.pipeline.Pipeline): sklearn fit model including 
        preprocessing pipeline and Logistic regression
    """
    model = load('final_model.joblib')
    return model

@app.get('/')
def get_root():
    """Defines the Home (root) behavior of API

    Returns:
        json: Welcome message
    """
    return {'message': 'Welcome to the failure detection API'}


@app.post("/predict")
async def predict(payload:InputFormulaire):
    """Defines the predict behavior of API

    Args:
        payload (InputFormulaire): user's input

    Returns:
        json: the model prediction :
        'prediction': prediction_class (0,1),
        'pred': prediction (fail, won't fail)
        'failure_proba': associated failure probability (from 0:  no probability to fail to 1: will fail)
    """
    # convert the payload to pandas DataFrame
    #input_df = pd.DataFrame(payload.dict(), index=range(1))
    df = prepare_input(payload)
    prediction_class = get_model().predict(df)[0]
    prediction_convert = {1:'fail', 0:'wont fail'}
    prediction = prediction_convert[prediction_class]
    predict_proba = get_model().predict_proba(df)[0][1]
    return {
        'prediction': prediction_class,
        'pred': prediction,
        'failure_proba': predict_proba
    }

