import math, random, time, os
import warnings
import pandas as pd
import numpy as np
import dask
import dask.array as da
import dask.dataframe as dd
import jax.numpy as jnp
from tqdm import tqdm

class daskIterativeFAMD(object):

    def __init__(self):
        dask.config.set({"optimization.fuse.ave-width": 5})
        self.explained_var = 0.95
        self.max_iter = 1000
        self.tol = 1e-4
        
    def calculateSVD(self, data):
        x = da.from_array(data.values, chunks='auto')#.to_dask_array(lengths=True)
        #U, s, VT = da.linalg.svd(x, coerce_signs=False) # 30.76, 37.89
        U, s, VT = da.linalg.svd_compressed(x, k=data.shape[0], compute=True, coerce_signs=False)
        return U, s, VT
        
    def encodeFeatures(self, dataframe, features_to_encode):
        # Maintain dataframe for columns that do not need to be encoded
        dropped = dataframe.drop(columns=features_to_encode)
        dropped_cols = dropped.columns.values
        # Create encoded feature dataframe. Take slices of this by column and reindex, joining to the master df
        # This will impute null values for encoded columns where necessary while maintaining encodings for other categorical columns
        feature_df = dataframe[features_to_encode]
        encoded_columns = []
        df_list = [dropped]
        
        for col in features_to_encode:
            sliced = feature_df[col].dropna()
            one_hot = pd.get_dummies(sliced, prefix=col+"_", sparse=True)
            if len(np.unique(sliced.values)) == 1:
                pass
            else:
                one_hot = one_hot.reindex(feature_df[col].index)
            encoded_columns.extend(one_hot.columns.values)
            df_list.append(one_hot)
            one_hot = None
        dropped = pd.concat(df_list, axis=1)
        return dropped, encoded_columns
    
    def initializeValues(self, dataframe):
        proportions = np.zeros(len(self.categorical))
        stdevs = np.zeros(len(self.continuous))

        # Sustitute the column means as the value for continuous nans, then standardize column
        # Substitute the missing values of categorical nans as the proportion of their category
        for index, c in enumerate(self.continuous):
            prop = dataframe[c].mean()
            if np.isnan(dataframe[c].values).any():
                dataframe[c] = np.where(dataframe[c].isnull(), prop, dataframe[c])
            stdev = np.std(dataframe[c])
            if stdev == 0:
                stdev = 0.01
            dataframe[c] = dataframe[c] / stdev
            stdevs[index] = stdev
        for index, c in enumerate(self.categorical):
            prop = np.abs(dataframe[c].mean())
            if np.isnan(dataframe[c].values).any():
                dataframe[c] = np.where(dataframe[c].isnull(), prop, dataframe[c])
            proportions[index] = prop
            dataframe[c] = dataframe[c] / np.sqrt(prop)        

        # Standardize by dividing each column by the square root of the column-wise proportion of each category
        variances = [s ** 2 for s in stdevs]
        variances.extend(proportions)
        diag = np.diag(variances)
        return dataframe, diag

    def famdImpute(self, dataframe):
        j = 0
        X = dask.persist(dataframe.copy())[0]
        #for i in range(self.max_iter):
        for i in tqdm(range(self.max_iter), position=0, leave=True):
            prev_df = X.copy()
            #print("----X----", X)
            new_df, diag = self.initializeValues(X)
            new_diag = diag.copy()
            for j in range(diag.shape[0]):
                if diag[j][j] != 0:
                    new_diag[j][j] = diag[j][j] ** (-1/2)
            XD = new_df.dot(new_diag)
            #XD = pd.DataFrame(np.dot(new_df, new_diag), index=X.index, columns=X.columns)
            M = np.mean(XD, axis=0)
            means = np.broadcast_to(M, XD.shape)
            resultant = (XD - means)#.to_numpy()#jnp.asarray((XD - means))
            U, s, VT = self.calculateSVD(resultant)
            n_elements = 0
            explained_var = da.cumsum((s ** 2)/ da.sum(s ** 2))
            for index, v in enumerate(explained_var):
                if v > self.explained_var:
                    n_elements = index
                    break
            U = U[:, :n_elements]
            sigma = np.diag(s[:n_elements])
            VT = VT[: n_elements]
            lr = da.dot(da.dot(U, sigma), VT)
            if self.noise == "gaussian":
                mu, sig = 0, da.std(lr, axis=0)
                for index, s in enumerate(sig):
                    noise = da.random.normal(mu, sig[index], lr.shape[0])
                    lr[:, index] += noise
            mul = (lr + means).dot(diag)
            for index, c in enumerate(X.columns):
                if da.isnan(dataframe[c].values).any():
                    replace_col = mul[:, index].compute()
                    X[c] = np.where(dataframe[c].isna(), replace_col, dataframe[c])
                else:
                    X[c] = dataframe[c]
            #print("---- X2 ----", X)
            diff = da.linalg.norm(X.values - prev_df.values)
            old = da.linalg.norm(prev_df.values)
            if (diff/old) < self.tol:
                break
        return X
    
    
    def fillNa(self, to_fill, imputed, categorical, exclude):
        for col in to_fill.columns.values:
            if col not in exclude:
                if col in categorical:
                    cols_to_max = [c for c in imputed.columns.values if c.startswith(col)]
                    to_fill[col] = imputed[cols_to_max].idxmax(axis=1).str.split("__").apply(lambda x : x[1]).astype(to_fill[col].dtype)
                else:
                    to_fill[col] = imputed[col]
            else:
                continue
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
            
    def impute(self, impute_df, dataframe, features_to_encode, exclude):
        impute_df = self.famdImpute(impute_df)
        dataframe = self.fillNa(dataframe, impute_df, features_to_encode, exclude)
        return dataframe
