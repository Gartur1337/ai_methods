import math
import matplotlib.pyplot as plt
import random
import pandas as pd


def new_index_list(arr) -> int:
    max_length = 0
    for inarr in arr:
        if len(inarr) > max_length:
            max_length = len(inarr)

    index_list = [i for i in range(max_length)]
    random.shuffle(index_list)
    return index_list

def shuffle_by_index_list(arr):
    index_list = new_index_list(arr)
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
    w = [random.uniform(0,1) for i in range(4)]
    data_set = [[] for i in range(4)]
    error_array = []
    arr = [[] for i in range(4)]
    v = 0.5
    excel_data = pd.read_excel('lab1.xlsx')

    data = pd.DataFrame(excel_data, columns=['x1', 'x2', 'x3', 'y'])

    data_set = [list(v) for k, v in data.items()]
    
    print(f'weights: {w}')

    for set in data_set:
        print(set)
 
    for i in range(len(data_set)):
        for j in range(len(data_set[i])):
            arr[i].append((data_set[i][j] - min(data_set[i])) / (max(data_set[i]) - min(data_set[i])))

    for set in arr:
        print(set,'\n')

    for _ in range(400):
        error = 0
        delta_arr = []
        for k in range(20):
            s = 0
            for i in range(len(w)):
                if i == 0:
                    print(f'w {i} {w[i]}  * 1')
                    s += w[i] * 1
                else:
                    print(f'w {i} {w[i]}  arr {arr[i-1][k]}')
                    s += w[i] * arr[i-1][k]
                    
            f = 1 / (1 + math.exp(-0.1 * s))
            print(f'f {f}')
            delta = arr[-1][k] - f
            print(f'delta {delta}')

            for i in range(len(w)):
                if i == 0:
                    w[i] +=  delta * v
                else:
                    w[i] += delta * v * arr[i-1][k]
            delta_arr.append(delta)
        
        for j in delta_arr:
            error += math.pow(j,2)

        error = math.sqrt(error / 20)
        
        error_array.append(error)
        arr = shuffle_by_index_list(arr)
        print(f'on epoch {_} error {error}')

    plt.plot(error_array)
    plt.show()

if __name__ == '__main__':
    main()