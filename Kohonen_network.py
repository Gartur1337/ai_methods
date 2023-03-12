import random
import math

class KohonenNetwork:
    def __init__(self, learning_rate=0.3, learning_rate_decrease=0.05, cluster_size=4):
        self.learning_rate = learning_rate
        self.learning_rate_decrease = learning_rate_decrease
        self.cluster_size = cluster_size
        self.weights =  []
        self.cluster_examples = [0] * cluster_size 
    
    def normalize_data(self, data):
        x_normalize = [[(data[j][i] - min(data[j])) / (max(data[j]) - min(data[j])) for i in range(len(data[j]))] for j in range(len(data))]

        return x_normalize


    def train(self, data):
        self.weights = [[0.20, 0.20, 0.30, 0.40, 0.40, 0.20, 0.50],
                         [0.20, 0.80, 0.70, 0.80, 0.70, 0.70, 0.80],
                         [0.80, 0.20, 0.50, 0.50, 0.40, 0.40, 0.40],
                         [0.80, 0.80, 0.60, 0.70, 0.70, 0.60, 0.70]]
        
        num_epochs = self.learning_rate / self.learning_rate_decrease
        
        for epoch in range(int(round(num_epochs))):
            random.shuffle(data)
            cluster_counts = [0] * self.cluster_size
            for data_set in range(len(data)):
                r = []  
                for k in range(self.cluster_size):
                    s = 0
                    for i in range(len(data[0])):
                        s += (data[data_set][i] - self.weights[k][i]) ** 2
                    r.append(math.sqrt(s))
                
                bmu = min(r)
                bmu_r = r.index(bmu)

                cluster_counts[bmu_r] += 1  

                for i in range(len(self.weights[bmu_r])):
                    self.weights[bmu_r][i] += self.learning_rate * (data[data_set][i] - self.weights[bmu_r][i])
            
            self.learning_rate -= self.learning_rate_decrease
        print(f'Epoch {epoch+1}, Cluster Counts: {cluster_counts}')
        

    def print_weights(self):
        for weight in self.weights:
            print(f'{weight}\n')

def main():
    data = [[1.00, 1.00, 0.17, 0.78, 0.70, 0.77, 0.68],
        [1.00, 0.00, 0.17, 0.58, 0.35, 0.00, 0.00],
        [0.00, 0.00, 0.17, 0.58, 0.35, 0.70, 0.60],
        [1.00, 1.00, 1.00, 0.77, 0.84, 0.75, 1.00],
        [0.00, 1.00, 0.33, 0.77, 0.70, 0.71, 0.71],
        [0.00, 1.00, 0.17, 0.77, 0.90, 0.87, 0.63],
        [0.00, 1.00, 0.00, 0.78, 0.65, 0.74, 0.81],
        [1.00, 0.00, 0.00, 0.52, 0.58, 0.59, 0.63],
        [1.00, 0.00, 0.00, 0.57, 0.24, 0.68, 0.49],
        [1.00, 0.00, 0.17, 0.52, 0.35, 0.13, 0.00],
        [0.00, 1.00, 1.00, 0.90, 0.99, 1.00, 1.00],
        [0.00, 1.00, 0.17, 0.89, 0.88, 0.70, 0.63],
        [1.00, 0.00, 0.00, 0.61, 0.00, 0.05, 0.49],
        [0.00, 1.00, 0.83, 0.83, 0.72, 0.77, 0.81],
        [1.00, 0.00, 0.00, 0.00, 0.03, 0.03, 0.49],
        [0.00, 1.00, 0.17, 0.65, 0.66, 0.68, 0.49],
        [1.00, 1.00, 0.67, 1.00, 1.00, 0.89, 1.00],
        [0.00, 1.00, 1.00, 0.85, 0.94, 0.92, 0.81],
        [1.00, 1.00, 0.83, 0.52, 0.58, 0.74, 0.49],
        [1.00, 0.00, 0.00, 0.57, 0.35, 0.03, 0.63]]

    kohonen_network = KohonenNetwork()
    kohonen_network.train(data)
    kohonen_network.print_weights()

if __name__ == '__main__':
    main()