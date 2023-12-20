import numpy as np
import pandas as pd

from Helpers.data import clean_dataset

data = clean_dataset.drop("vowel", axis=1)
coeffs = np.corrcoef(data.to_numpy(), rowvar=False)
coeffs_table = pd.DataFrame(coeffs, columns=data.columns, index=data.columns)

style= coeffs_table.style.format(precision=4, decimal=".").format(precision=4)

print(style.to_latex())