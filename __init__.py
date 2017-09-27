import gzip
from collections import defaultdict
import random
import numpy as np
from sklearn.metrics import mean_squared_error
from dask.array.random import beta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="blue", color_codes=True)


    
path = "../amazon-fine-foods/"
# training_all=[]
# for l in readGz(path+"finefoods.txt.gz"):
#     training_all.append(l)
#     
# print "done"

food = pd.read_csv(path+"Reviews.csv",header=0)
# food.head()
print food["Score"].value_counts()
