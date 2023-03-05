import math
import random
import pandas as pd
import math
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        self.w = [random.uniform(0, 1) for i in range(4)]
        self.v = 0.5
        self.data_set = [[] for i in range(4)]
        self.error_array = []
        self.arr = [[] for i in range(4)]

    def read_data(self, filename):
        excel_data = pd.read_excel(filename)
        data = pd.DataFrame(excel_data, columns=['x1', 'x2', 'x3', 'y'])
        self.data_set = [list(v) for k, v in data.items()]

    def normalize_data(self):
        for i in range(len(self.data_set)):
            for j in range(len(self.data_set[i])):
                self.arr[i].append((self.data_set[i][j] - min(self.data_set[i])) / (max(self.data_set[i]) - min(self.data_set[i])))
        print(self.arr)

    def activation_function(self, s):
        return 1 / (1 + math.exp(-0.1 * s))

    def calculate_error(self, delta_arr):
        error = 0
        for j in delta_arr:
            error += math.pow(j, 2)
        return math.sqrt(error / 20)

    def train(self, num_epochs):
        for _ in range(num_epochs):
            error = 0
            delta_arr = []
            for k in range(20):
                s = 0
                for i in range(len(self.w)):
                    if i == 0:
                        s += self.w[i] * 1
                    else:
                        s += self.w[i] * self.arr[i - 1][k]
                f = self.activation_function(s)
                delta = self.arr[-1][k] - f

                for i in range(len(self.w)):
                    if i == 0:
                        self.w[i] += delta * self.v
                    else:
                        self.w[i] += delta * self.v * self.arr[i - 1][k]
                delta_arr.append(delta)

            error = self.calculate_error(delta_arr)
            self.error_array.append(error)
            self.arr = self.shuffle_by_index_list(self.arr)
            print(f'on epoch {_} error {error}')

    def shuffle_by_index_list(self, arr):
        max_length = 0
        for inarr in arr:
            if len(inarr) > max_length:
                max_length = len(inarr)

        index_list = [i for i in range(max_length)]
        random.shuffle(index_list)
        new_arr = []
        for i in arr:
            new_arr.append([])

        for index in index_list:
            inside_index = 0
            for i in new_arr:
                i.append(arr[inside_index][index])
                inside_index += 1

        return new_arr

def main():
    nn = NeuralNetwork()
    nn.read_data('lab1.xlsx')
    nn.normalize_data()
    nn.train(400)

    plt.plot(nn.error_array)
    plt.show()


if __name__ == '__main__':
    main()