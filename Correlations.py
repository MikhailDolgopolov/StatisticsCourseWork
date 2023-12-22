import numpy as np
import pandas as pd

from Helpers.data import clean_dataset, vowels, pivot_vowels
data=pivot_vowels(clean_dataset)
# data = clean_dataset.drop("vowel", axis=1)
# data = pivot_vowels(clean_dataset[['vowel']])
print(data)
coeffs = np.corrcoef(data.to_numpy(), rowvar=False)
coeffs_table = pd.DataFrame(coeffs, columns=data.columns, index=data.columns)

print(coeffs_table)

style= coeffs_table.style.format(precision=4, decimal=".").format(precision=4)

print(style.to_latex())