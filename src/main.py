from src.models import Model
import numpy as np
import matplotlib.pyplot as plt


m = Model(data_path='/home/weiss/workspace/studies/Supervised Learning/project/projet_app_sup_20/data/Aggregation.txt',
          sep='\t')
m = Model(data_path='/home/weiss/workspace/studies/Supervised Learning/project/projet_app_sup_20/data/creditcard.csv',
          sep=',')
m = Model(data_path='/home/weiss/workspace/studies/Supervised Learning/project/projet_app_sup_20/data/flame.txt',
          sep='\t')
m = Model(data_path='/home/weiss/workspace/studies/Supervised Learning/project/projet_app_sup_20/data/spiral.txt',
          sep='\t')
m = Model(data_path='/home/weiss/workspace/studies/Supervised Learning/project/projet_app_sup_20/data/VisaPremier.txt',
          sep='\t',
          ys=-2)

b = m._check_balance()
plt.bar(range(len(b)),b.values(),align='center')
plt.xticks(range(len(b)), list(b.keys()))
plt.show()