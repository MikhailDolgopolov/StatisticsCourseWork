from matplotlib import pyplot as plt

image_path = "Output/Images/"
def latex_to_png(latex_str, name):
    fig = plt.figure()
    plt.axis("off")
    plt.text(0.5, 0.5, f"${latex_str}$", size=50, ha="center", va="center")
    png_path = f"{image_path}{name}.png"

    plt.savefig(png_path, format = "png", bbox_inches="tight", pad_inches = 0.4)
    plt.close(fig)

def latex_to_tex(latex_str, name):
    file = open(f"Output/Latex/{name}.tex", "w")
    file.write(latex_str)
    file.close()

import numpy as np
import matplotlib.pyplot as plt
def draw_equation(latex_str):
    plt.clf()
    plt.axis("off")
    plt.text(0.5, 0.5, f"${latex_str}$", size=1000/len(latex_str), ha="center", va="center")
    plt.show(block=True)

class LinRegEquation:
    def __init__(self, model):
        self.intercept = model.intercept_
        self.coeffs = model.coef_.squeeze()

    def get_result(self, factors, rounding=3):
        if len(factors)!=len(self.coeffs):
            raise ValueError("Wrong number of factors")
        equation = ""
        intercept = round(self.intercept, rounding)
        coefs = np.round(self.coeffs, rounding)
        if intercept!=0:
            equation+=str(intercept)
        for i in range(len(self.coeffs)):
            if coefs[i]==0: continue
            equation+= f"{'' if coefs[i]<0 else '+'}{str(round(coefs[i], rounding))}*{str(factors[i])}"
        if intercept==0:
            equation = equation[1:]
        return equation
