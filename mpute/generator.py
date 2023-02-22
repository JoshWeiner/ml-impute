import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from .customThread import CustomThread
from .iterativeFAMD import iterativeFAMD
from .daskIterativeFAMD import daskIterativeFAMD

class Generator(object):
    
    def __init__(self):
        self.engine = 'default'

    def generate(self, dataframe, encode_cols=[], exclude_cols=[], max_iter=1000, tol = 1e-3, explained_var = 0.95, method="single", n_versions = 20, noise="gaussian", engine='default'):
        gen = iterativeFAMD()

        if engine == 'dask':
            gen = daskIterativeFAMD()
            self.engine = 'dask'
        
        gen.explained_var = explained_var
        gen.max_iter = max_iter
        gen.tol = tol

        if method != "single" and method != "multiple" and method is not None:
            raise ValueError("Method of imputation must be single or multiple.")
        else:
            if method is not None:
                gen.method = method

            if gen.method == "single":
                gen.n_versions = 1
                gen.noise = None
            else:
                gen.n_versions = n_versions
                if noise != "gaussian":
                    raise ValueError("Please specify a valid method of adding noise to generated data")
                else:
                    gen.noise = noise
    
        if len(encode_cols) == 0:
            encode_cols = dataframe.columns[(dataframe.dtypes=='object') | (dataframe.dtypes=='category')].tolist()
        features_to_encode, drop_cols = gen.filterColumns(encode_cols, exclude_cols)
        impute_df = dataframe.copy().reset_index().drop(columns=np.array(drop_cols))
        impute_df, categorical = gen.encodeFeatures(impute_df, features_to_encode)
        gen.categorical = categorical
        gen.continuous = impute_df.columns.difference(categorical)
        if gen.method == "single":
            impute_df = gen.famdImpute(impute_df)
            dataframe = gen.fillNa(dataframe.copy(), impute_df, features_to_encode, drop_cols)
            return dataframe
        elif self.engine != 'dask':
            pool = Pool(processes=n_versions)
            df_dict = {}
            gen.threads = []
            results = []
            data = (impute_df.copy(), dataframe.copy(), features_to_encode, drop_cols, )
            for i in range(gen.n_versions):
                result = pool.apply_async(gen.impute, data)
                results.append(result)
            pool.close()
            #pool.join()
            ready_count = 0
            while True:
                #time.sleep(1)
                try:
                    ready = [result.ready() for result in results]
                    successful = [result.successful() for result in results]
                except Exception:
                    continue
                if all(successful):
                    break
                if all(ready) and not all(successful):
                    raise Exception(f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')
                else:
                    print(np.sum(ready) / n_versions)
            for i, res in enumerate(results):
                df_dict[i] = res.get()
            return df_dict
        else:
            gen.threads = []
            df_dict = {}
            data = (impute_df.copy(), dataframe.copy(), features_to_encode, drop_cols, )
            for i in range(n_versions):
                t = CustomThread(target=gen.impute, args=data)
                t.start()
                gen.threads.append(t)
            for i, thread in enumerate(gen.threads):
                df_dict[i] = thread.join()
            return df_dict
            

