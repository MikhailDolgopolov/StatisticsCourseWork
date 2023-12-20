import numpy as np
from Helpers.data import clean_dataset
class ModelData:
    def __init__(self, model, samples, r_=0):
        self.general_samples = clean_dataset.shape[0]
        self.__model = model
        self.__r_sq = r_
        self.__sample_size = samples/self.general_samples
        self.__name = "Unnamed model"

    def name(self, string=""):
        if string=="":
            return self.__name
        self.__name=string
    @property
    def model(self):
        return self.__model
    @property
    def r_sq(self):
        return self.__r_sq
    @property
    def sample_size(self):
        return self.__sample_size
    @property
    def weighted_rsq(self):
        return self.r_sq*self.sample_size

    def __str__(self):
        return f"Model '{self.__name}' from {round(self.general_samples*self.__sample_size)} samples"

    @classmethod
    def score(cls,model, X, Y):
        return model.score(X,Y)

    def predict(self, X):
        X = np.array(X)

        return self.__model.predict(X)
    @classmethod
    def custom_score(cls,model, X, Y):
        Y_pred = model.predict(X)
        # print(f"Predicted:  \n {Y_pred}")
        # print(f"Rounded:  \n {np.round(Y_pred)}")
        Y_pred = np.round(Y_pred)
        Rs = Y-Y_pred

        SSR = sum([r**2 for r in Rs])
        Y_mean = np.mean(Y)
        SST = sum([(Yi - Y_mean)**2 for Yi in Y])
        return 1-SSR/SST
