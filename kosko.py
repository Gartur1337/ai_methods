import numpy as np

class KoskoNetwork:
    
    def __init__(self) -> None:
        self.weights = np.array([])

    def train(self, input_data, output_data) -> np.ndarray:
        self.weights = np.dot(input_data.T, output_data)
        return self.weights

    def predict(self, pattern) -> np.ndarray:
        outputs = []
        old_output = np.zeros_like(pattern)
        iterations = 0
        while True:
            iterations += 1
            output = np.sign(np.dot(self.weights.T, pattern))
            output = np.sign(np.dot(self.weights, output))
            outputs.append(output.flatten())
            if np.allclose(output, old_output):
                break
            pattern = output
            old_output = output
        print(f'number of iterations {iterations}')
        return outputs[-1]

def main():
    input_data = np.array(([1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1],
                           [1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,1]))
    output_data = np.array(([1,1,1,1,-1,1,1,-1,1],
                            [1,-1,1,1,1,-1,1,-1,1]))
    Kn = KoskoNetwork()
    tr = Kn.train(input_data,output_data)
    print(tr)
    print(Kn.predict([1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,-1,-1,1]))
    print(Kn.predict([-1,-1,1,1,1,-1,1,1,1,-1,-1,1,1,1,1,1]))

if __name__ == '__main__':
    main()