from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression

from Helpers.ModelData import ModelData
from Helpers.data import clean_dataset, pivot_vowels

from models import test_model

import pandas as pd
import numpy as np
from pprint import pprint

def create_model(dataset, output):
    dataset.dropna(axis=0, inplace=True)
    y = dataset[output].to_numpy()
    if len(y)<2:
        return ModelData(None, 0)
    m_parameters = dataset.drop(columns=[output])
    x = m_parameters.to_numpy()

    model = LinearRegression().fit(x, y)

    r_sq = ModelData.score(model, x, y)
    print(f"Standard score: {ModelData.score(model, x, y)}")
    model_data = ModelData(model, len(y), r_sq)
    return model_data


data = pivot_vowels(clean_dataset)
print(data.columns)
def predict_end(dataset):
    min_length = np.min(dataset["length"])
    max_length = np.max(dataset["length"])
    steps = 50
    delta = (max_length-min_length)/(steps+1)
    for i in range(1, steps+1):
        dataset[f"less than {round(min_length+delta*i)}"] = np.where(dataset["length"]<min_length+delta*i, 1,0)
    print(dataset)
    dataset["length"] = dataset["length"] * 0.01
    dataset["2nd power"] = dataset["length"] ** 2
    # 0.498

    dataset["3rf power"] = dataset["length"] ** 3
    dataset["4th power"] = dataset["length"] ** 4
    dataset["4th power"] = dataset["length"] ** 4
    dataset["sqrt"] = dataset["length"] ** 0.5
    # 0.516
    dataset["length:rhymes"] = dataset["length"] * dataset["rhymes"]
    dataset["length:syl"] = dataset["length"] * dataset["syl"]
    dataset["length:closed"] = dataset["length"] * dataset["closed"]
    # 0.523
    # syl, rhymes, closed
    dataset["syl:rhymes"] = dataset["syl"] * dataset["rhymes"]
    dataset["syl:closed"] = dataset["syl"] * dataset["closed"]
    dataset["rhymes:closed"] = dataset["rhymes"] * dataset["closed"]
    # 0.529
    return dataset

def predict_length(dataset):
    dataset["syl:rhymes"] = dataset["syl"] * dataset["rhymes"]
    dataset["syl:closed"] = dataset["syl"] * dataset["closed"]
    dataset["syl:end"] = dataset["syl"] * dataset["closed"]
    dataset["rhymes:closed"] = dataset["rhymes"] * dataset["closed"]
    dataset["rhymes:end"] = dataset["rhymes"] * dataset["end"]
    dataset["closed:end"] = dataset["closed"] * dataset["end"]
    #0.521
    return dataset

output = "end"
my_model = create_model(predict_end(data), output)


