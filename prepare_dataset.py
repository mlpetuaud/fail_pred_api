import os
import datetime
import pandas as pd
import numpy as np
import datetime

"""
    This file is used to perform transformation on an input dataset (DataFrame)
    in order to use the returned DataFrame into the failure prediction model pipeline
    used in the API (app.py file)
"""

def add_custom_cols(df):

    """This function adds some calculated columns to the dataset df

    Args:
        df (pandas DataFrame): dataframe holding the future dataset table : companies
        accounts for the 2018 year

    Returns:
        pandas DataFrame: dataframe with added columns
    """

    df["Credit client"] = (df['Clients et comptes rattachés (3) (net) (BXNET) 2018 (€)']*365)/(df["Chiffre d'affaires net (Total) (FL) 2018 (€)"]*1.2)

    df["Credit Fournisseurs"] = df['Dettes fournisseurs et comptes rattachés (DX) 2018 (€)']*365/(
        (df['Achats de marchandises (y compris droits de douane) (FS) 2018 (€)'] 
        + df['Achats de matières premières et autres approvisionnements (y compris droits de douane) (FU) 2018 (€)'] 
        + df['Autres achats et charges externes (3) (6 bis) (FW) 2018 (€)']
        )*1.2)

    df["Rotation_stocks"] = ( df['Matières premières, approvisionnements (net) (BLNET) 2018 (€)'] + 
             + df['En cours de production de biens (net) (BNNET) 2018 (€)'] 
             + df['En cours de production de services (net) (BPNET) 2018 (€)']
             + df['Produits intermédiaires et finis (net) (BRNET) 2018 (€)']
             + df['Marchandises (net) (BTNET) 2018 (€)'])*365 / (
                df["Chiffre d'affaires net (Total) (FL) 2018 (€)"]  
                - df["1 - RESULTAT D'EXPLOITATION (I - II) (GG) 2018 (€)"])

    df["BFR"] = (
        df['TOTAL (III) (net) (CJNET) 2018 (€)']
        + df['Valeurs mobilières de placement (net) (CDNET) 2018 (€)']
        + df['Disponibilités (net) (CFNET) 2018 (€)']
        - df['Avances et acomptes reçus sur commandes en cours (DW) 2018 (€)']
        - df['Dettes fournisseurs et comptes rattachés (DX) 2018 (€)']
        - df['Dettes fiscales et sociales (DY) 2018 (€)']
        - df['Dettes sur immobilisations et comptes rattachés (DZ) 2018 (€)']
        - df['Autres dettes (EA) 2018 (€)']
        - df["Produits constatés d'avance (EB) 2018 (€)"])


    df["BFRE"] = (
                df['Matières premières, approvisionnements (net) (BLNET) 2018 (€)']
                 + df['En cours de production de services (net) (BPNET) 2018 (€)']
                 + df['En cours de production de biens (net) (BNNET) 2018 (€)']
                 + df['Produits intermédiaires et finis (net) (BRNET) 2018 (€)']
                 + df['Marchandises (net) (BTNET) 2018 (€)']
                 + df['Avances et acomptes versés sur commandes (net) (BVNET) 2018 (€)']
                 + df['Clients et comptes rattachés (3) (net) (BXNET) 2018 (€)']
                - df['Avances et acomptes reçus sur commandes en cours (DW) 2018 (€)']
                - df['Dettes fournisseurs et comptes rattachés (DX) 2018 (€)']
                - df['Dettes fiscales et sociales (DY) 2018 (€)']
                - df['Autres dettes (EA) 2018 (€)']
                 )

    df["Endettement total"] = (df['Autres emprunts obligataires (DT) 2018 (€)']
                 + df['Emprunts obligataires convertibles (DS) 2018 (€)']
                 + df['Emprunts et dettes auprès des établissements de crédit (5) (DU) 2018 (€)']
                 + df['Emprunts et dettes financières divers (DV) 2018 (€)']
                - df['(5)\xa0Dont concours bancaires courants, et soldes créditeurs de banques et CCP (EH) 2018 (€)'])

    df["CAF"] = (df['3 - RESULTAT COURANT AVANT IMPOTS (I - II + III - IV + V - VI) (GW) 2018 (€)']
                - df['Reprises sur amortissements et provisions, transferts de charges (9) (FP) 2018 (€)']
                + df["Dotations d'exploitation sur immobilisations (dotations aux amortissements) (GA) 2018 (€)"]
                + df["Dotations d'exploitation sur immobilisations (dotations aux provisions) (GB) 2018 (€)"]
                + df["Dotations d'exploitation sur actif circulant (dotations aux provisions) (GC) 2018 (€)"]
                + df["Dotations d'exploitation pour risques et charges (dotations aux provisions) (GD) 2018 (€)"]
                - df['Reprises sur provisions & transferts de charges (GM) 2018 (€)']
                + df['Dotations financières aux amortissements et provisions (GQ) 2018 (€)']
                - df["Participation des salariés aux résultats de l'entreprise (HJ) 2018 (€)"]
                - df['Impôts sur les bénéfices (HK) 2018 (€)'])

    df["Capacite de remboursement"] = df["Endettement total"] / df["CAF"]

    df["Ressources durables"] = (df['TOTAL (I) (DL) 2018 (€)']
                 + df['TOTAL(II) (DO) 2018 (€)']
                 + df['TOTAL (III) (DR) 2018 (€)']
                 + df['Autres emprunts obligataires (DT) 2018 (€)']
                 + df['Emprunts obligataires convertibles (DS) 2018 (€)']
                 + df['Emprunts et dettes auprès des établissements de crédit (5) (DU) 2018 (€)']
                 + df['Emprunts et dettes financières divers (DV) 2018 (€)']
                 - df['(5)\xa0Dont concours bancaires courants, et soldes créditeurs de banques et CCP (EH) 2018 (€)']
                 - df['Capital souscrit non appelé (I) (AA) 2018 (€)'])

    df["FRNG"] = (df["Ressources durables"]   
                 + df['Ecarts de conversion passif (V) (ED) 2018 (€)']
                 - df['Primes de remboursement des obligations (CM) 2018 (€)']
                 - df['Ecarts de conversion actif (CN) 2018 (€)']
                 + df['TOTAL (II) (net) (BJNET) 2018 (€)'])

    df["Taux endettement"] = df["Endettement total"] / df["Ressources durables"]

    df["Rentabilite financiere"] = (df["RESULTAT DE L'EXERCICE (bénéfice ou perte) (DI) 2018 (€)"] 
                                    / (df['TOTAL (I) (DL) 2018 (€)'] 
                                       - df['Capital souscrit non appelé (I) (AA) 2018 (€)']))

    df["EBE"] = (df["Chiffre d'affaires net (Total) (FL) 2018 (€)"]
                + df["Subventions d'exploitation (FO) 2018 (€)"]
                + df['Production stockée (FM) 2018 (€)']
                + df['Production immobilisée (FN) 2018 (€)']
                - df['Achats de marchandises (y compris droits de douane) (FS) 2018 (€)']
                - df['Variation de stock (marchandises) (FT) 2018 (€)']
                - df['Achats de matières premières et autres approvisionnements (y compris droits de douane) (FU) 2018 (€)']
                - df['Variation de stock (matières premières et approvisionnements) (FV) 2018 (€)']
                - df['Autres achats et charges externes (3) (6 bis) (FW) 2018 (€)']
                - df['Impôts, taxes et versements assimilés (FX) 2018 (€)']
                - df['Salaires et traitements (FY) 2018 (€)']
                - df['Charges sociales (10) (FZ) 2018 (€)']
                + df['(3)\xa0Dont Crédit-bail mobilier (HP) 2018 (€)']
                + df['(3)\xa0Dont Crédit-bail immobilier (HQ) 2018 (€)']
                )

    df["VA"] = (df["EBE"] 
                - df["Subventions d'exploitation (FO) 2018 (€)"]
                + df['Impôts, taxes et versements assimilés (FX) 2018 (€)']
                + df['Salaires et traitements (FY) 2018 (€)']
                + df['Charges sociales (10) (FZ) 2018 (€)'])

    df["Liquidite generale"] = ((df['TOTAL (III) (net) (CJNET) 2018 (€)']
                               - df["Charges constatées d'avance (3) (net) (CHNET) 2018 (€)"])
                               / (df['Avances et acomptes reçus sur commandes en cours (DW) 2018 (€)']
                                 + df['Dettes fournisseurs et comptes rattachés (DX) 2018 (€)']
                                 + df['Dettes fiscales et sociales (DY) 2018 (€)']
                                 + df['Dettes sur immobilisations et comptes rattachés (DZ) 2018 (€)']
                                 + df['Autres dettes (EA) 2018 (€)']
                                 + df['(5)\xa0Dont concours bancaires courants, et soldes créditeurs de banques et CCP (EH) 2018 (€)']))


    df["Liquidite reduite"] = (
        (
            (df['TOTAL (III) (net) (CJNET) 2018 (€)']
                               - df["Charges constatées d'avance (3) (net) (CHNET) 2018 (€)"]
            )
                               - (
                                   df['Matières premières, approvisionnements (net) (BLNET) 2018 (€)']
                                 + df['En cours de production de services (net) (BPNET) 2018 (€)']
                                 + df['En cours de production de biens (net) (BNNET) 2018 (€)']
                                 + df['Produits intermédiaires et finis (net) (BRNET) 2018 (€)']
                                 + df['Marchandises (net) (BTNET) 2018 (€)']
                                 )
        ) 
                                / (
                                    df['Avances et acomptes reçus sur commandes en cours (DW) 2018 (€)']
                                 + df['Dettes fournisseurs et comptes rattachés (DX) 2018 (€)']
                                 + df['Dettes fiscales et sociales (DY) 2018 (€)']
                                 + df['Dettes sur immobilisations et comptes rattachés (DZ) 2018 (€)']
                                 + df['Autres dettes (EA) 2018 (€)']
                                 + df['(5)\xa0Dont concours bancaires courants, et soldes créditeurs de banques et CCP (EH) 2018 (€)']
                                    )
                                )

    df["Taux ressources propres"] = ((df['TOTAL (I) (DL) 2018 (€)'] 
                                     - df['Capital souscrit non appelé (I) (AA) 2018 (€)'])
                                    / df['TOTAL GENERAL (I à V) (EE) 2018 (€)'])


    df["Rentabilite des capitaux propres"] = (df["RESULTAT DE L'EXERCICE (bénéfice ou perte) (DI) 2018 (€)"]
                                             / (df['TOTAL (I) (DL) 2018 (€)']
                                                + df['TOTAL(II) (DO) 2018 (€)']
                                                - df['Capital souscrit non appelé (I) (AA) 2018 (€)']))

    df["Autonomie financiere"] = ((df['TOTAL (I) (DL) 2018 (€)']
                                                + df['TOTAL(II) (DO) 2018 (€)']
                                                - df['Capital souscrit non appelé (I) (AA) 2018 (€)'])
                                  / df['TOTAL GENERAL (I à V) (EE) 2018 (€)'])

    df["Poids interets"] = (df['Intérêts et charges assimilées (GR) 2018 (€)'] 
                            / df["1 - RESULTAT D'EXPLOITATION (I - II) (GG) 2018 (€)"])

    df["Taux EBE"] = (df["EBE"]
                      / df["Chiffre d'affaires net (Total) (FL) 2018 (€)"])

    df["Taux VA"] = (df["VA"]
                      / df["Chiffre d'affaires net (Total) (FL) 2018 (€)"])

    df["Taux Rentabilite"] = (df["RESULTAT DE L'EXERCICE (bénéfice ou perte) (DI) 2018 (€)"]
                             / df["Chiffre d'affaires net (Total) (FL) 2018 (€)"])

    df["Poids dettes fiscales"] = ((df['Sécurité sociale et autres organismes sociaux - Montant brut (8D) 2018 (€)']
                                   + df['Impôts sur les bénéfices - Montant brut (8E) 2018 (€)']
                                   + df['T.V.A. - Montant brut (VW) 2018 (€)'])
                                   / df["Chiffre d'affaires net (Total) (FL) 2018 (€)"])

    df["Tresorerie"] = df["FRNG"] - df["BFR"]
    
    df["Taux augmentation endettement CT"] = (df['Emprunts souscrits en cours d’exercice - à 1 an au plus (VJ2) 2018 (€)']
                                         / df['TOTAL GENERAL (I à V) (EE) 2018 (€)'])
                      
    df["Date de création"] = pd.to_datetime(df["Date de création"])
    df["Age entreprise"] = ((datetime.datetime(2019,12,31) - df["Date de création"])/np.timedelta64(1, 'M'))
    df["Age entreprise"] = df["Age entreprise"].astype(int)
    
    return df


def apply_categorical_dtypes(df):
    """This function converts 2 columns (Catégorie juridique (Niveau II) and a_21) of the input DataFrame into category dtypes

    Args:
        df (pandas.DataFrame) holding the dataset

    Returns:
        df (pandas.DataFrame) holding the dataset
    """
    df = df.astype({"Catégorie juridique (Niveau II)" : 'category', 
                            'a_21' : 'category'})
    return df

def merge_naf_v2(df, df_naf):
    """This function performs a left join between dataset and a table holding a sectorial mapping table
        in order to change sectorial granularity

    Args:
        df (pandas.DataFrame): dataframe holding the dataset
        df_naf (pandas.DataFrame): dataframe holding a mapping table between different sectorial granularity levels

    Returns:
        df (pandas.DataFrame): merged df
    """
    merged_df = pd.merge(df, df_naf, how = 'left', left_on = df['Code APE'], right_on = df_naf['code_ape'])
    merged_df = merged_df.drop(['key_0'], axis=1)
    assert merged_df.shape[0]==df.shape[0]
    assert merged_df.shape[1]==df.shape[1]+df_naf.shape[1]
    return merged_df

def remove_useless_cols_api(df):
    """This function drops some columns from DataFrame that are not relevant for a failure prediction purpose

    Args:
        df (pandas.DataFrame): dataframe holding the dataset

    Returns:
        df (pandas.DataFrame): dataframe holding the dataset
    """
    useless_cols = ["Code APE",
        'index',
        'code_ape',
        'descriptif_code_ape',
        'a_615',
        'a_272',
        'a_129',
        'a_88',
        'a_64',
        'a_38',
        'a_10',
        'Unnamed: 0',
        "Emprunts souscrits en cours d’exercice - à 1 an au plus (VJ2) 2018 (€)"]
    df = df.drop(useless_cols, axis=1)
    return df

#### MAIN ####

def prepare_dataset_api(df):
    """This function is the main function of this file.
        It performs the relevant transformations to an input DataFrame 
        in order to use the returned DataFrame as an input for the failure prediction model

    Args:
        df (pandas.DataFrame): DataFrame required with 88 columns (see app.InputFormulaire) :

    Returns:
        pandas.DataFrame: transformed DataFrame with 115 columns ready to be used for prediction by model
    """
    #input = 88 cols
    df = add_custom_cols(df) # + 28 cols = 116 cols
    df_naf = pd.read_csv("naf.csv", delimiter=";")
    df = merge_naf_v2(df, df_naf) # adds 11 cols => 127 cols
    df = remove_useless_cols_api(df) # removes 12 cols => 115 cols (excluding target)
    df = apply_categorical_dtypes(df)
    assert df.shape[1]==115
    return df
