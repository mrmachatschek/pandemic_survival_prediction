import pandas as pd 
import numpy as np 
from boruta import BorutaPy
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.feature_selection import SequentialFeatureSelector

def get_features(train_set, target, method=None, model="rf", n_features="auto", verbose=1):
        if model == "rf":
            model = RandomForestClassifier(n_jobs=-1, random_state=1)
        elif model == "gb":
            model = GradientBoostingClassifier(random_state=1)
        
        if method == None: 
            selected_features = train_set.columns.values
                
        if method == "boruta":    
            print("Fitting Boruta...")    
            boruta = BorutaPy(model, n_estimators=n_features, verbose=verbose)
            boruta.fit(train_set.values, target.values)
            selected_features = train_set.columns[boruta.support_].values
            
        if method == "rfe":
            print("Fitting Recursive Feature Elimination...") 
            rfe = RFECV(estimator=model, cv=4, scoring='accuracy', verbose=verbose)
            rfe = rfe.fit(train_set, target)
            selected_features = train_set.columns[rfe.support_].values
        
        if method == "sfs":
            print("Fitting Sequential Feature Selection...")
            if n_features == "auto":
                n_features = "best"
            sfs = SequentialFeatureSelector(model, k_features=n_features, verbose=verbose, n_jobs=-1, scoring='accuracy', cv=4)
            sfs.fit(train_set, target)
            selected_features = list(sfs.k_feature_names_)

        return selected_features