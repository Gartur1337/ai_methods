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
    print(w,'\n')

    excel_data = pd.read_excel('lab1.xlsx')

    data = pd.DataFrame(excel_data, columns=['x1', 'x2', 'x3', 'y'])

    data_set = [list(v) for k, v in data.items()]

    for i in range(len(data_set)):
        for j in range(len(data_set[i])):
            normalization_data_set[i].append((data_set[i][j] - min(data_set[i])) / (max(data_set[i]) - min(data_set[i])))

    for set in normalization_data_set:
        print(set,'\n')

    for _ in range(20):
        s = 0
        delta = 0
        v = 0.5
        ans = []
        delta_array = []

        for k in range(14):
            for i in range(len(w)):
                if i == 0:
                    s += w[i] * 1
                else:
                    s += w[i] * normalization_data_set[i][k]

            f = 1 / (1 + math.exp(-0.1 * s))

            delta = normalization_data_set[-1][k] - f
    
            for i in range(len(w)):
                if i == 0:
                    w[i] += delta * v
                else:
                    w[i] += delta * v * normalization_data_set[i][k]      
            
            ans.append(f)
            delta_array.append(delta)

    print("answer ", ans, '\n'*2, "delta", delta_array)    

if __name__ == '__main__':
    main()