import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def load_keywords(path):
    keywords = []
    return keywords


def load_attention_weights(path):
    attention_weights = []
    return attention_weights


plt.figure(figsize=(10, 6))
att = [[0, 0, 0.89, 0, 0.03, 0.08, 0, 0, 0, 0],
       [0.01, 0.69, 0, 0.14, 0, 0.16, 0, 0, 0, 0],
       [0, 0, 0.39, 0.58, 0, 0.02, 0, 0, 0, 0]]
x = pd.DataFrame(att, index=['S1', 'S2', 'S3'],
                 columns=['performance', 'speed', 'memory', 'time', 'resource', 'limit', 'requirement', 'performing', 'meet', 'function'])
sns.heatmap(x, annot=True, fmt='.2f', cmap='RdPu', vmin=0, vmax=1)
plt.xticks(rotation=45)
plt.yticks(rotation=360)
plt.show()