import itertools

from pandas.core.frame import DataFrame

import pandas as pd
import numpy as np
from pprint import pprint
from models import create_model
from Helpers.data import pivot_vowels, clean_dataset


def split_data(dataset, columns):
    x = np.array([len(dataset[c].unique()) for c in columns])

    c = [[np.full(x[ic], columns[ic]), dataset[columns[ic]].unique()] for ic in range(len(columns))]

    matches = [[{columns[i]: c[i][1][xi]} for xi in range(x[i])] for i in range(len(columns))]
    multi_dicts = cartesian(matches)

    dicts = [{k: v for d in L for k, v in d.items()} for L in multi_dicts]

    queries = [' and '.join([f'{k} == {repr(v)}' for k, v in d.items()]) for d in dicts]

    data = [dataset.query(q) for q in queries]

    data = [data[i].drop(columns=dicts[i].keys()) for i in range(len(data))]

    # print(data[0].columns)
    return data, dicts


def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


columns = clean_dataset.columns
columns = np.array(columns.drop(["length", "end"]))

nested = [list(itertools.combinations(columns, r+1)) for r in range(len(columns))]
combos = list(itertools.chain.from_iterable(nested))

split_results = [(split_data(clean_dataset, split)) for split in combos]
models = [[create_model(pivot_vowels(bundle[0][i]))
            for i in range(len(bundle[1]))]
            for bundle in split_results]


[[models[j][i].name(' '.join([f'{k}={repr(v)}' for k, v in split_results[j][1][i].items()]))
            for i in range(len(split_results[j][1]))]
            for j in range(len(split_results))]

results = {", ".join(split_results[k][1][0].keys())
           :sum([models[k][i].weighted_rsq for i in range(len(models[k]))])
            for k in range(len(models))}

results = pd.Series(results).sort_values(ascending=False)[:6]
print("----  Weighted average of R_sq of several models:")
print(results)
print()

models = list(itertools.chain.from_iterable(models))

ind_models_w = {str(m): m.weighted_rsq for m in models}
ind_models = {str(m): m.r_sq for m in models}

ind_results = pd.Series(ind_models).sort_values(ascending=False)[:15]

print("Absolute R_sq of several models:")
print(ind_results)
print()
ind_weighted_results = pd.Series(ind_models_w).sort_values(ascending=False)[:15]
print("R_sq relative to the fraction of dataset explained: ")
print(ind_weighted_results)
