import pandas as pd
import numpy as np
import re

clean_dataset = pd.read_csv("Data/CleanDataset.csv")
vowels = ["a", "o", "u", "e", "i", "y"]

def digitize_original():
    full_dataset = pd.read_csv("Data/FullDataset.csv")
    print(full_dataset)
    full_dataset.to_csv("Data/FullDataset.csv", index = False)
    full_dataset['rhymes'] = np.where(full_dataset['genre'] == "проза", 0, 1)

    cl_dataset = full_dataset.drop(columns=["genre"])
    cl_dataset.to_csv("Data/CleanDataset.csv", index=False)


def pivot_vowels(dataset, save=False):
    if "vowel" not in dataset.columns:
        return dataset
    dataset["ones"] = 1
    vowel_pivot = dataset.pivot(columns="vowel", values="ones")
    dataset = dataset.drop(columns="ones")
    if dataset.shape[0] < 20:
        for v in vowels:
            if v not in vowel_pivot.columns:
                dataset[v] = 0
    vowel_pivot = vowel_pivot.fillna(0)
    pruned_data = dataset.drop(columns=["vowel"])
    pivoted_data = pd.concat([pruned_data, vowel_pivot], axis=1)

    if save:
        pivoted_data.to_csv("Data/VowelPivotedDataset.csv", index=False)
    return pivoted_data


def sample_table(name, n):
    df = pd.read_csv(f"Data/{name}.csv")
    sample = pd.concat([df[:n], df[-n:]], axis=0)
    sample = sample.style.format(precision=1, decimal=".").format(precision=1)

    print(sample.to_latex())

# sample_table("VowelPivotedDataset", 8)