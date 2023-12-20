import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from Helpers.ModelData import ModelData
from Helpers.data import clean_dataset


def pivot_v_types(dataset, save=False):
    dataset["high"] = [1 if x in ["i", "u", "y"] else 0 for x in dataset["vowel"]]
    dataset["middle"] = [1 if x in ["a", "y"] else 0 for x in dataset["vowel"]]
    dataset["back"] = [1 if x in ["o", "u"] else 0 for x in dataset["vowel"]]
    dataset = dataset.drop(columns="vowel")
    if save:
        dataset.to_csv("Data/TraitPivotedDataset.csv", index=False)
    return dataset




def create_model(dataset):
    dataset.dropna(axis=0, inplace=True)
    y = dataset["length"].to_numpy()
    if len(y)<2:
        return ModelData(None, 0,0)
    parameters = dataset.drop(columns=["length"])
    x = parameters.to_numpy()

    model = LinearRegression().fit(x, y)

    r_sq = ModelData.score(model, x, y)
    model_data = ModelData(model, len(y), r_sq)
    return model_data


def predict(model, row):
    X = np.array(row).reshape(1, -1)
    Y = X[0][0]
    X = [X[0][1:]]
    answer = round(model.predict(X)[0], 1)
    return answer


def random_row(restrictions={}, n=1):
    data = clean_dataset.drop(columns="length").drop_duplicates()
    query = ' and '.join([f'{k} == {repr(v)}' for k, v in restrictions.items()])
    requested = data if query == "" else data.query(query)
    df = pd.DataFrame({"length": np.random.normal(160, 40, size=n),
                       "vowel": np.random.choice(requested["vowel"], size=n),
                       "syl": np.random.choice(requested["syl"], size=n),
                       "end": np.random.choice(requested["end"], size=n),
                       "closed": np.random.choice(requested["closed"], size=n),
                       "rhymes": np.random.choice(requested["rhymes"], size=n),
                       })
    return df


def test_model(model, dataset, given_columns, transformer):
    # np.random.seed(0)
    query = ' and '.join([f'{k} == {repr(v)}' for k, v in given_columns.items()])
    new_df = dataset if query == "" else dataset.query(query)
    lengths = new_df["length"].to_numpy().squeeze()

    import matplotlib.pyplot as plt
    hist_values, edges, _ = plt.hist(lengths)
    bin_width = edges[1] - edges[0]

    def sample_predictions(n):
        predictions = []
        for i in range(n):
            # row = transformer(new_df.sample())
            row = transformer(random_row(given_columns))
            print(row)
            prediction = predict(model, row)

            predictions.append(prediction)

        return predictions

    model_answers = sample_predictions(round(len(lengths) * 0.5))
    plt.hist(model_answers, bins=round((np.max(model_answers)-np.min(model_answers))/bin_width+1), alpha=0.5)
    plt.savefig(f'Output/Images/LM_RE{"".join(map(str, given_columns.values()))}.png', bbox_inches='tight')
    plt.show(block=True)





#Критерий Краскала-Уоллиса для построения адекватных подвыборок

#Критерий Манна-Уитни для оценки качества модели в более узком случае подвыборок

#Построение моделей по подвыборкам с меньшим количеством параметров
