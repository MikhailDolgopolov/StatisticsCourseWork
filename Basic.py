import numpy as np
import matplotlib.pyplot as plt
from Helpers.data import clean_dataset
import seaborn as sns

g = sns.FacetGrid(clean_dataset, col="vowel",  row="end")
g.map_dataframe(sns.histplot, x="length")
plt.show(block=True)