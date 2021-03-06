# -*- coding: utf-8 -*-
"""
Functions used to find happymeal product categories:
 - Exclus
 - Viande, œufs
 - Poisson
 - Produits gras sucrés salés
 - Matières grasses ajoutées
 - Produits laitiers (hors fromage)
 - Fromage
 - Féculents raffinés
 - Féculents non raffinés
 - Fruits
 - Légumes
 - Plats préparés
"""

__author__ = 'Mehdi Miah'
__license__ = 'MIT License'
__version__ = '0.1'
__maintainer__ = 'Julie Seguela'
__status__ = 'Development'


import sys
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import itertools
import openfoodfacts


def get_foodGroup(EAN, 
                  Produit_Nom, 
                  convert_groups,
                  model_classifier, 
                  model_matching): 
    '''
    -- Input --
    EAN : EAN code, string
    Produit_Nom : name of the product, string
    convert_groups : dictionnary which enables to create a family of food from OpenFoodFacts' groups
    model_classifier : model which predicts the foodgroups from nutrients
    model_matching : model which predicts the foodgroups from names
    
    -- Output --
    food_group : the group of the product, string
    statut : how the foodgroup has been obtained
        1 - the product is in OFF and belongs to a well defined foodgroup 
        2 - the product is in OFF and its foodgroup is predicted from nutrients
        3 - the product is not in OFF and its foodgroup is predicted from its name
        
    -- Examples --
    get_foodGroup(EAN = "4260436322114", Produit_Nom = None)
    get_foodGroup(EAN = "5410233710105", Produit_Nom = None)
    get_foodGroup(EAN = "hgbjnklhgc", Produit_Nom = "Pizza")
    '''
    
    try : #incase of missing EAN
        product_off = openfoodfacts.products.get_product(str(EAN))  #gets the product from Open Food Facts
    except :
        pass

    try : #manages to get info on pnns_groups_2
        
        product_off_groups2 = product_off['product']['pnns_groups_2']
    
        if product_off_groups2 in convert_groups.keys() : #if the product of OFF belongs to a well defined group
            foodgroup = convert_groups[product_off_groups2]
            statut = 1
            
            return [foodgroup, statut]
    except : pass    
        
    try : #manages to get info on nutriments

        #looks for nutrients
        df_nutrients = pd.DataFrame([product_off['product']['nutriments']],
                               dtype = 'float64')[['salt_100g', 'fat_100g', 'sugars_100g', 'proteins_100g', 'carbohydrates_100g', 'saturated-fat_100g']]

        # We will predict if and only if the values are valid
        df_nutrients = df_nutrients[df_nutrients['salt_100g'] <= 100]
        df_nutrients = df_nutrients[df_nutrients['sugars_100g'] <= 100]
        df_nutrients = df_nutrients[df_nutrients['carbohydrates_100g'] <= 100]
        df_nutrients = df_nutrients[df_nutrients['fat_100g'] <= 100]
        df_nutrients = df_nutrients[df_nutrients['proteins_100g'] <= 100]
        df_nutrients = df_nutrients[df_nutrients['saturated-fat_100g'] <= 100]

        n_row = df_nutrients.shape[0] #1 if values are correct, 0 if one value over 100

        if n_row == 1 : #no missing values and no weird values

            #then predicts the foodgroup from nutrients
            foodgroup = model_classifier.predict(df_nutrients[['salt_100g', 'sugars_100g',
                                                               'carbohydrates_100g', 'fat_100g',
                                                               'proteins_100g', 'saturated-fat_100g']])[0]
            statut = 2
            
            return [foodgroup, statut]
    except : 
        pass
    
    try : #manages to predicts the foodgroup from the name
        foodgroup = model_matching.predict([Produit_Nom])[0]
        statut = 3
        return [foodgroup, statut]

    except : #arggg
        return [None, None]



def get_foodGroupFromToDF(listing_df, 
                          EAN_col,
                          product_name_col,
                          mapping_file,
                          model_classifier_file,
                          model_matching_file): 

    '''
    -- Input --
    listing_df : listing of food products we want to put in balanced meals, as dataframe
                 (contains at least EAN_col and product_name_col)
    EAN_col : column containing EAN code, as string
    product_name_col : column containing the product name, as string
    mapping_file : path of file which enables to map OpenFoodFacts' groups to our food groups
    model_classifier_file : path of file containing model which predicts the food groups from nutrients
    model_matching_file : path of file containing model which predicts the food groups from names
    
    -- Output --
    listing_df : the same dataframe, with 2 columns added
        labelAlim_2 : food group for balanced meals
        statutAlim_2: how the foodgroup has been obtained
        
    -- Example --
    get_foodGroupFromToDF(listing_df            = input_listing, 
                          EAN_col               = 'EAN',
                          product_name_col      = 'Produit_Nom',
                          mapping_file          = 'data/mapping_off_ideal.csv',
                          model_classifier_file = 'data/clf_nutrients_rf_groupeAlim_2_light.sav',
                          model_matching_file   = 'data/clf_names_nb_light.sav')
    '''
    
    # Check if listing_df contains EAN_col and product_name_col
    if pd.Series([EAN_col, product_name_col]).isin(listing_df.columns).sum() < 2:
        sys.exit(EAN_col + ' or ' + product_name_col + ' is not in dataframe')
   
    else:
        # Model to get the foodgroup of a product which is in the Open Food Facts database
        clf_nutrients_rf = pickle.load(open(model_classifier_file, 'rb'))
        
        # Model to get the foodgroup of a product which is not in the Open Food Facts database
        clf_names_nb = pickle.load(open(model_matching_file, 'rb'))
        
        # Mapping file
        mapping_groups = pd.read_csv(mapping_file, sep = ';', encoding = 'UTF-8')
        
        # Transform into a dictionnary
        dict_mapping = mapping_groups.set_index('pnns_groups_2')['groupeAlim_2'].to_dict()
        
        # Add food group
        listing_df[['labelAlim_2', 'statutAlim_2']] = \
            listing_df.apply(lambda row: get_foodGroup(row[EAN_col], row[product_name_col],
                                                       dict_mapping, clf_nutrients_rf, clf_names_nb), 
                             axis=1, result_type='expand')
        
        return listing_df
 