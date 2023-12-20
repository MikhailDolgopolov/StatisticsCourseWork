
from models import create_model, test_model
from Helpers.data import pivot_vowels, clean_dataset
import matplotlib.pyplot as plt
import numpy as np
from Helpers.latex import LinRegEquation, latex_to_png

transform = pivot_vowels
data = transform(clean_dataset)
columns = list(data.columns.drop("length"))
my_model = create_model(data).model

print(my_model.intercept_)
print(LinRegEquation(my_model).get_result(columns, 1))


test_model(my_model, clean_dataset, { "rhymes":1, "end":0}, transform)
# test_model(my_model, clean_dataset, {"rhymes":1, "end":1}, transform)
# test_model(my_model, clean_dataset, {"rhymes":0, "end":0}, transform)
# test_model(my_model, clean_dataset, {"rhymes":0, "end":1}, transform)
