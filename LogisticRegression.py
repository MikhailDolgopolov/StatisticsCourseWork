import numpy as np

from Helpers.data import clean_dataset, pivot_vowels

seed = 162
np.random.seed(seed)
data = pivot_vowels(clean_dataset)
# data = data.drop("rhymes", axis=1)
print(data.shape)
#split dataset in features and target variable
feature_cols = data.columns.drop("end")
print(f"{len(feature_cols)} параметров")
X = data[feature_cols] # Features
y = data.end # Target variable


from sklearn import metrics
import numpy as np

def logistic(d):
    latex_log = r"\frac{1}{End} = 1+exp("
    for key, value in d.items():
        if "int" in key.lower():
            latex_log += f"{value}+"
        else:
            latex_log += f"{key}*{'('+str(value)+')' if (value<0) else value}+"
    return latex_log[:-1] + ")"

import statsmodels.api as sm
sm_model = sm.Logit(y, X).fit()


def sm_params(smodel):
    keys = ["intercept", *feature_cols]
    values = np.round(np.array(smodel.params.values, dtype=np.float64), 4)
    return dict(zip(keys, values))

print()
print(f"SM model: {logistic(sm_params(sm_model))}")
print()
print(sm_model.summary())


datum = data.drop("end", axis=1)
y_pred = np.where(sm_model.predict(datum)>0.42, 1,0)

cm = metrics.confusion_matrix(data["end"], y_pred)

precision = cm[0,0]/(cm[0,0]+cm[1,0])
recall = cm[0,0]/(cm[0,0]+cm[0,1])
accuracy = (cm[0,0]+cm[1,1])/np.sum(cm)

print("Precision: ", precision)
print("Recall: ", recall)
print("Accuracy: ", accuracy)
print("F1", 2*precision*recall/(precision+recall))



