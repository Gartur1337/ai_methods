import numpy as np
import time

class HopfieldNetwork:
    def __init__(self, n):
        self.n = n
        self.weights = np.zeros((n, n))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.reshape(pattern, (self.n, 1))
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)        

    def predict(self, pattern, max_iterations=1000):
        pattern = np.reshape(pattern, (self.n, 1))
        output = np.dot(self.weights, pattern)
        output[output >= 0] = 1
        output[output < 0] = -1
        prev_output = output.flatten()
        last_outputs = [prev_output]
        iter_num = 0
        while iter_num < max_iterations:
            iter_num += 1
            pattern = output.flatten()
            output = np.dot(self.weights, pattern)
            output[output >= 0] = 1
            output[output < 0] = -1
            if np.array_equal(output.flatten(), prev_output):
                break
            if any([np.array_equal(output.flatten(), x) for x in last_outputs]):
                break
            last_outputs.append(output.flatten())
            if len(last_outputs) > 10:
                last_outputs.pop(0)
            prev_output = output.flatten()
            print(output.flatten())
            time.sleep(1)
        return output.flatten()


def main():
    n = 4
    patterns = np.array([[1, 1, -1, -1], [-1, -1, 1, 1], [1, -1, 1, -1]])
    hn = HopfieldNetwork(n)
    hn.train(patterns)
    print(hn.predict([1, 1, 1, -1]))  # [-1, -1, 1, -1]

if __name__ == '__main__':
    main()