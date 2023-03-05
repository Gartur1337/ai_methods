import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def main():

    excel_data = pd.read_excel('lab2.xlsx')
    arr_x =  [[] for i in range(3)]
    arr_y =  []

    data_x = pd.DataFrame(excel_data, columns=['x1', 'x2', 'x3']) 
    data_y = pd.DataFrame(excel_data, columns=['y']) 

    data_set_x = [list(v) for k, v in data_x.items()]
    data_set_y= [list(v) for k, v in data_y.items()]

    for i in range(len(data_set_x)):
        for j in range(len(data_set_x[i])):
            arr_x[i].append((data_set_x[i][j] - min(data_set_x[i])) / (max(data_set_x[i]) - min(data_set_x[i])))

    for i in range(len(data_set_y)):
        for j in range(len(data_set_y[i])):
            arr_y.append((data_set_y[i][j] - min(data_set_y[i])) / (max(data_set_y[i]) - min(data_set_y[i])))

    arr = []

    for num in arr_y:
        num = [num]
        arr.append(num)
    y = np.array(arr)

    x = np.array([list(row) for row in zip(*arr_x)])

    

    
    epochs = 1

    learning_rate = 0.9

    input_layer_size = 3
    hidden_layer_size = 2
    output_layer_size = 1

    hidden_weights = np.random.uniform(size=(input_layer_size, hidden_layer_size))
    print(f' hidden_weights \n {hidden_weights}')
    output_weights = np.random.uniform(size=(hidden_layer_size, output_layer_size))
    print(f'output_weights \n {output_weights}')

    errors = []
    for epoch in range(epochs):
        
        hidden_layer_activation = np.dot(x, hidden_weights)
        print(f'hidden_layer_activation \n{hidden_layer_activation}')
        hidden_layer_output = sigmoid(hidden_layer_activation)
        print(f'hidden_layer_output \n {hidden_layer_output}')
        output_layer_activation = np.dot(hidden_layer_output, output_weights)
        print(f'output_layer_activation \n {output_layer_activation}')
        predicted_output = sigmoid(output_layer_activation)
        print(f'predicted_output \n {predicted_output}')
    
        error = y - predicted_output
        print(f'error \n {error}')
        output_delta = error * sigmoid_derivative(predicted_output)
        print(f'output_delta \n {output_delta}')
        hidden_error = output_delta.dot(output_weights.T)
        print(f'hidden_error \n {hidden_error}')
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)
        print(f'hidden_delta \n {hidden_delta}')


        output_weights += hidden_layer_output.T.dot(output_delta) * learning_rate
        hidden_weights += x.T.dot(hidden_delta) * learning_rate

        epoch_error = np.mean(np.abs(error))
        print(f'On epoch: {epoch} Error equal: {epoch_error}')
        errors.append(epoch_error)

    print(f'Predicted values: \n {predicted_output}')

    plt.plot(errors)
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.title('Зависимость ошибки от эпохи')
    plt.show()

if __name__ == '__main__':
    main()