# Формула
# h = sin(x1) * (x2 + x3) 


import pandas as pd
import math
import random

def activate_func(x1,x2,x3):
    func = math.sin(x1) * (x2 + x3) 

def main():
    x1, x2, x3, y = [], [], [], []

    w = [random.uniform(0,1) for i in range(4)]
    print(w)

    excel_data = pd.read_excel('lab1.xlsx')

    data = pd.DataFrame(excel_data, columns=['x1', 'x2', 'x3', 'y'])

    x0 = 1    
    x1.extend(data['x1'])
    x2.extend(data['x2'])
    x3.extend(data['x3'])
    y.extend(data['y'])

    x1_normalization = []
    x2_normalization = []
    x3_normalization = []
    y_normalization = []

    for i in range(len(x1)):
        x1_normalization.append((x1[i] - min(x1)) / (max(x1) - min(x1)))
        x2_normalization.append((x2[i] - min(x2)) / (max(x2) - min(x2)))
        x3_normalization.append((x3[i] - min(x3)) / (max(x3) - min(x3)))
        y_normalization.append((y[i] - min(y)) / (max(y) - min(y)))

    print(x1_normalization,'\n', x2_normalization,'\n', x3_normalization,'\n', y_normalization,'\n')

    for _ in range(10000):
        s = 0
        delta = 0
        v = 0.5
        ans = []
        for k in range(len(x1_normalization)):
            for i in range(len(w)):
                if i == 0:
                    s += w[i] * x0
                if i == 1:
                    s += w[i] * x1_normalization[k]
                if i == 2:
                    s += w[i] * x2_normalization[k]
                if i == 3:
                    s += w[i] * x3_normalization[k]

            f = 1 / (1 + math.exp(-0.001 * s))

            delta = y_normalization[k] - f

            for i in range(len(w)):
                if i == 0:
                    w[i] += delta * v
                if i == 1:
                    w[i] += delta * v * x1_normalization[k]
                if i == 2:
                     w[i] += delta * v * x2_normalization[k]
                if i == 3:
                     w[i] += delta * v * x3_normalization[k]

            ans.append(delta)

    print(ans)    

if __name__ == '__main__':
    main()