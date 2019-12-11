import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.stats as sch

df_raw = pd.read_json(r"data.json")
data = df_raw[['age','countGroups']]


X = np.array(data)

test = sch.pearsonr(X)
print(test)