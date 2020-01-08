import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

class Model:
    def __init__(self, data_path, mode='csv', sep=',',xs: tuple = (0, -2), ys: int = -1):
        if (mode == 'csv'):
            self.data = pd.read_csv(data_path, delimiter=sep)
        else:
            print('NO OTHER MODES SUPPORTED YET')
        self._split_data(xs,ys)
    def _split_data(self, xs: tuple = (0, -2), ys: int = -1):
        self.X = self.data.values[:, xs[0]:xs[1]]
        self.Y = self.data.values[:, ys]

    def _check_balance(self):
        us = set(self.Y)
        balances = {u:len([i for i in self.Y if i == u])/self.X.shape[0] for u in us}
        print(balances)
        return  balances

if __name__ == '__main__':
    pass