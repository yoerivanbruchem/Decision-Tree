from random import randint
from math import floor
import pandas as pd


class DataToolSet:

    def __init__(self, data):
        self.__data = pd.DataFrame(data)
        self.test_data = pd.DataFrame()
        self.train_data = pd.DataFrame(data)

    def split_data(self, perc_of_test_set=20):
        data_count = len(self.__data)
        percentage_of_test_set = perc_of_test_set
        length_of_test_set = floor(data_count * percentage_of_test_set / 100)

        train_set = self.__data
        test_set = pd.DataFrame()

        for x in range(length_of_test_set):
            indices = train_set.index.values.tolist()
            index = indices[randint(0, len(indices) - 1)]
            row = train_set.loc[[index]]
            test_set = test_set.append(row, ignore_index=True)
            train_set = train_set.drop(index)

        self.train_data = train_set
        self.test_data = test_set
