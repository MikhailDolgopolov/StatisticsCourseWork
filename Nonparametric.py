from pprint import pprint

import numpy as np
import pandas as pd
import scipy

from Helpers.data import clean_dataset, vowels
import matplotlib.pyplot as plt


# grouping = ["a", "ou", "eiy"]
# grouping = [[*ch] for ch in grouping]

def general_pvalue(groups, comparing="end"):
    qs = [' and '.join([f'{k} == {repr(v)}' for k, v in {"vowel": arr}.items()]) for arr in
          groups]

    dts = [clean_dataset.query(q)[["length", "vowel", comparing]] for q in qs]
    p_sum = 0
    variants = list(clean_dataset[comparing].unique())

    for i in range(len(groups)):
        middle = dts[i][dts[i][comparing] == variants[0]]["length"]
        end = dts[i][dts[i][comparing] == variants[1]]["length"]

        mann = scipy.stats.mannwhitneyu(middle, end)
        p_sum += mann.pvalue
    return p_sum / len(groups)


def split(a):
    if not a:
        return [[]]
    elif len(a) == 1:
        return [[a]]
    else:
        result = []
        for i in range(1, len(a) + 1):
            result += [(a[:i], *sub_split) for sub_split in split(a[i:])]
        return result


def display_partition(grouping, p_val, comp="end", num=0):
    queries = [' and '.join([f'{k} == {repr(v)}' for k, v in {"vowel": arr for set in arr}.items()]) for arr in
               grouping]

    data = [clean_dataset.query(q)[["length", "vowel", comp]] for q in queries]
    fig = plt.figure()
    fig.set_figheight(7)
    gs = fig.add_gridspec(len(grouping), 1, hspace=0.25, wspace=0)
    plots = gs.subplots(sharex='col')
    print(f"p_value here is {p_val}")
    # fig.suptitle(("" if num == 0 else f"{num}th best, ") + f"Differences in {comp},\n p_value: {p_val}")
    # fig.suptitle(f"p_value: {p_val}")
    variants = list(clean_dataset[comp].unique())
    if len(grouping) == 1:
        middle = data[0][data[0][comp] == variants[0]]["length"]
        end = data[0][data[0][comp] == variants[1]]["length"]
        a, htype = 1, "step"
        plots.hist(middle, alpha=a, histtype=htype, linewidth=3, label="end: 0")
        plots.hist(end, alpha=a, histtype=htype, linewidth=3, label="end: 1")
        plots.legend(["end: 0", "end: 1"])
    fig.savefig("Output/Images/WM_A.png", bbox_inches="tight")
    plt.show(block=True)


partitions = split(vowels)


class Res:
    def __init__(self, array, p_value):
        self.array = [*array]
        self.p_value = p_value

    def __repr__(self):
        return f"{self.array} :   {self.p_value}"


def metric(res):
    return res.p_value


k = 5
p_results = sorted([Res(d, general_pvalue(d)) for d in partitions], key=metric)[:k][::-1]
pprint(p_results)
#
# for i in range(len(p_results)):
#     r=p_results[i]
# display_partition(r.array, r.p_value, k-i)

grouping = ["aoueiy"]
grouping = [[*ch] for ch in grouping]

other_cols = clean_dataset.columns.drop(["length", "vowel"])

# for c in other_cols:
c="end"
gp = general_pvalue(grouping, c)
display_partition(grouping, gp, c)
print(f"Probability of samples being the same on {c}: {round(gp,6)}")
