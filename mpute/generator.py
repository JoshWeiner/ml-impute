import numpy as np
import pandas as pd
from .customThread import CustomThread
from .iterativeFAMD import iterativeFAMD

class Generator(iterativeFAMD):
    
    def __init__(self):
        super(Generator, self).__init__()

    def generate(self, dataframe, encode_cols=[], exclude_cols=[], max_iter=1000, tol = 1e-4, explained_var = 0.95, method="single", n_versions = 20, noise="gaussian"):
        self.explained_var = explained_var
        self.max_iter = max_iter
        self.tol = tol

        if method != "single" and method != "multiple" and method is not None:
            raise ValueError("Method of imputation must be single or multiple.")
        else:
            if method is not None:
                self.method = method

            if self.method == "single":
                self.n_versions = 1
                self.noise = None
            else:
                self.n_versions = n_versions
                if noise != "gaussian":
                    raise ValueError("Please specify a valid method of adding noise to generated data")
                else:
                    self.noise = noise
    
        if len(encode_cols) == 0:
            encode_cols = dataframe.columns[(dataframe.dtypes=='object') | (dataframe.dtypes=='category')].tolist()
        features_to_encode, drop_cols = self.filterColumns(encode_cols, exclude_cols)
        impute_df = dataframe.copy().reset_index().drop(columns=np.array(drop_cols))
        impute_df, categorical = self.encodeFeatures(impute_df, features_to_encode)
        self.categorical = categorical
        self.continuous = impute_df.columns.difference(categorical)
        if self.method == "single":
            impute_df = self.famdImpute(impute_df)
            dataframe = self.fillNa(dataframe.copy(), impute_df, features_to_encode)
            return dataframe
        else:
            df_dict = {}
            self.threads = []
            for i in range(self.n_versions):
                t = CustomThread(target = self.impute, args=(impute_df.copy(), dataframe.copy(), features_to_encode))
                self.threads.append(t)
                t.start()
            for i, thread in enumerate(self.threads):
                df_dict[i] = thread.join()
            return df_dict
            

