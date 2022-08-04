import sklearn.utils.extmath as extmath
import math
import pandas as pd
import numpy as np
import random
import jax.numpy as jnp
import scipy.linalg as la
from tqdm import tqdm
from numpy.linalg import multi_dot
import time

class iterativeFAMD:
        
    def calculateDiagonal(self, dataframe):
        stdevs = np.std(dataframe[self.continuous]).replace(0, 0.1)
        proportions = dataframe[self.categorical].abs().mean().replace(0, 0.1)
        variances = stdevs ** 2
        diag = pd.concat([variances, proportions])
        diag = pd.DataFrame(np.diag(diag),index=diag.index,columns=diag.index)
        return diag
        
    def encodeFeatures(self, dataframe, features_to_encode):
        # Maintain dataframe for columns that do not need to be encoded
        dropped = dataframe.drop(columns=features_to_encode)
        dropped_cols = dropped.columns.values
        # Create encoded feature dataframe. Take slices of this by column and reindex, joining to the master df
        # This will impute null values for encoded columns where necessary while maintaining encodings for other categorical columns
        feature_df = dataframe[features_to_encode]
        encoded_columns = []
        df_list = [dropped]
        
        for col in tqdm(features_to_encode, position=0, leave=True):
            sliced = feature_df[col].dropna()
            one_hot = pd.get_dummies(sliced, prefix=col+"_", sparse=True)
            #print(one_hot.shape)
            if len(np.unique(sliced.values)) == 1:
                #print(one_hot)
                pass
            else:
                #print(col, one_hot.shape)
                one_hot = one_hot.reindex(feature_df[col].index)
            encoded_columns.extend(one_hot.columns.values)
            df_list.append(one_hot)
            one_hot = None
        dropped = pd.concat(df_list, axis=1)
        return dropped, encoded_columns
    
    def initializeValues(self, dataframe):
        # Get columns of continuous variables
        proportions = np.zeros(len(self.categorical))
        stdevs = np.zeros(len(self.continuous))

        # Sustitute the column means as the value for continuous nans, then standardize column
        for index, c in enumerate(self.continuous):
            prop = dataframe[c].mean()
            if np.isnan(dataframe[c].values).any():
                dataframe[c] = np.where(dataframe[c].isnull(), prop, dataframe[c])
            stdev = np.std(dataframe[c])
            if stdev == 0:
                #print(c, dataframe[c].value_counts(ascending=False) )
                stdev = 0.01#pass
                #raise Exception(f'Column [{c}] has zero variance!')
            else:
                dataframe[c] = dataframe[c] / stdev
            stdevs[index] = stdev
        for index, c in enumerate(self.categorical):
            prop = dataframe[c].mean()
            if np.isnan(dataframe[c].values).any():
                dataframe[c] = np.where(dataframe[c].isnull(), prop, dataframe[c])
            proportions[index] = prop
            dataframe[c] = dataframe[c] / np.sqrt(prop)        

        # Substitute the missing values of categorical nans as the proportion of their category
        # Standardize by dividing each column by the square root of the column-wise proportion of each category

        variances = [s ** 2 for s in stdevs]
        variances.extend(proportions)
        diag = np.diag(variances)
        return dataframe, diag

    def famdImpute(self, dataframe):
        j = 0
        diag = None
        X = dataframe.copy()
        prev_diff = np.infty
        for i in tqdm(range(self.max_iter), position=0, leave=False):
            prev_df = X.copy()
            new_df, diag = self.initializeValues(X)
            new_diag = diag.copy()
            for j in range(diag.shape[0]):
                if diag[j][j] != 0:
                    new_diag[j][j] = diag[j][j] ** (-1/2)
            XD = new_df.dot(new_diag)
            #XD = pd.DataFrame(np.dot(new_df, new_diag), index=X.index, columns=X.columns)
            M = np.mean(XD, axis=0)
            means = np.broadcast_to(M, XD.shape)
            resultant = (XD - means).to_numpy()
            U, s, VT = jnp.linalg.svd(resultant, full_matrices=False)
            n_elements = 0
            explained_var = np.cumsum((s**2)/np.sum(s**2))
            for index, v in enumerate(explained_var):
                if v > self.explained_var:
                    n_elements = index
                    break
            U = U[:, :n_elements]
            sigma = np.diag(s[:n_elements])
            VT = VT[: n_elements]
            lr = np.dot(np.dot(U, sigma), VT)
            mul = (lr + means).dot(diag)
            
            for index, c in enumerate(X.columns):
                if np.isnan(dataframe[c].values).any():
                    X[c] = np.where(dataframe[c].isnull(), mul[:, index], dataframe[c])
                else:
                    X[c] = dataframe[c]
            diff = np.sqrt(((X.values - prev_df.values)**2).sum())
            old = np.sqrt(((prev_df.values)**2).sum())
            if (diff/old) < self.tol:
                break
        return X
    
    
    def fillNa(self, to_fill, imputed, categorical):
        for col in to_fill.columns.values:
            if col in categorical:
                cols_to_max = [c for c in imputed.columns.values if c.startswith(col)]
                to_fill[col] = imputed[cols_to_max].idxmax(axis=1).str.split("__").apply(lambda x : x[1])
            else:
                to_fill[col] = imputed[col]
        return to_fill
    
    def filterColumns(self, features_to_encode, exclude):
        new_features = []
        drop_cols = [e for e in exclude]
        date_col_substrings = ["dt", "date", "_date"]
        for col in features_to_encode:
            if any(substring in col.lower() for substring in date_col_substrings) or col in exclude:
                drop_cols.append(col)
                continue
            else:
                new_features.append(col)
        if len(drop_cols) == 0:
            drop_cols = ["index"]
        else:
            drop_cols.append("index")
        return new_features, drop_cols
            
        

    def impute(self, dataframe, encode_cols=[], exclude_cols=[], max_iter=1000, tol = 1e-4, explained_var = 0.95):
        self.tol = tol
        self.explained_var = explained_var
        self.max_iter = max_iter
        if len(encode_cols) == 0:
            encode_cols = dataframe.columns[(dataframe.dtypes=='object') | (dataframe.dtypes=='category')].tolist()
        features_to_encode, drop_cols = self.filterColumns(encode_cols, exclude_cols)
        impute_df = dataframe.copy().reset_index().drop(columns=np.array(drop_cols))
        # Create encoded dataframe with initial values for nan values
        #impute_df[features_to_encode] = impute_df[features_to_encode].astype(object)
        impute_df, categorical = self.encodeFeatures(impute_df, features_to_encode)
        self.categorical = categorical
        self.continuous = impute_df.columns.difference(categorical)
        # Perform single value decomposition
        impute_df = self.famdImpute(impute_df)
        dataframe = self.fillNa(dataframe, impute_df, features_to_encode)
        return dataframe
