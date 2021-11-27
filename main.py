import pandas as pd
import nltk
import numpy as np
import tensorflow.keras as keras
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt

cafe = pd.read_csv('data/cafe.csv', index_col=0)
print(cafe)  # [500 rows x 13 columns]



