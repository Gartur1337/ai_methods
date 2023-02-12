import pandas as pd
import math
import random
from random import shuffle

def activate_func(x1,x2,x3):
    func = math.sin(x1) * (x2 + x3) 

def main():
    w = [random.uniform(0,1) for i in range(4)]
    data_set = [[] for i in range(4)]
    normalization_data_set = [[] for i in range(4)]

    print('weigths:', w, '\n')

    excel_data = pd.read_excel('lab1.xlsx')

    data = pd.DataFrame(excel_data, columns=['x1', 'x2', 'x3', 'y'])

    data_set = [list(v) for k, v in data.items()]
    
    for i in range(len(data_set)):
        for j in range(len(data_set[i])):
            normalization_data_set[i].append((data_set[i][j] - min(data_set[i])) / (max(data_set[i]) - min(data_set[i])))

    for set in normalization_data_set:
        print(set,'\n')

    for epoch in range(200):
        s = 0
        delta = 0
        v = 0.9
        ans = []
        delta_array = []
        result = 0
        for k in range(14):
            for i in range(len(w)):
                if i == 0:
                    s += w[i] * 1
                else:
                    s += w[i] * normalization_data_set[i-1][k]

            f = 1 / (1 + math.exp(-1 * s))
            delta = normalization_data_set[-1][k] - f

            for i in range(len(w)):
                if i == 0:
                    # print('weight:', i, w[i], '\n')
                    w[i] += delta * v
                else:
                    # print('weight:', i, w[i], '\n', 'data set:', normalization_data_set[i-1][k])
                    w[i] += delta * v * normalization_data_set[i-1][k]      
            ans.append(f)
            delta_array.append(delta)

        for k in range(14):
            normalization_data_set[-1][k] = math.sin(normalization_data_set[0][k]) * (normalization_data_set[1][k] + normalization_data_set[2][k])

        for i in range(len(normalization_data_set) - 1):
            shuffle(normalization_data_set[i])

        result += math.pow(normalization_data_set[-1][k] - f,2)
        e = math.sqrt(result * (1 / 14))
        print("Error on epoch: ", epoch, e)

    # print("answer ", ans, '\n'*2, "delta", delta_array)    

if __name__ == '__main__':
    main()