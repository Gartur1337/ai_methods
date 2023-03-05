import math

class ART_1:
    def __init__(self, x, r_crit=0.8, learning_rate=0.5, learning_rate_decrease=0.05, num_epochs=None, LAMBDA=2.0):
        self.x = x
        self.r_crit = r_crit
        self.learning_rate = learning_rate
        self.learning_rate_decrease = learning_rate_decrease
        self.num_epochs = num_epochs if num_epochs else int(round(learning_rate / learning_rate_decrease))
        self.LAMBDA = LAMBDA
        self.x_normalize = self.normalize_data(x)
        self.w = [[(LAMBDA * self.x_normalize[0][i]) / (LAMBDA - 1 + sum(self.x_normalize[0])) for i in range(len(self.x_normalize[0]))]]
        self.t = [[self.x[0][i] for i in range(len(self.x[0]))]]
    
    def normalize_data(self, data):
        x_normalize = [[] for i in range(len(data))]
        for i in range(len(data)):
            for j in range(len(data[i])):
                x_normalize[i].append((data[i][j] - min(data[i])) / (max(data[i]) - min(data[i])))
        return x_normalize
    
    def train(self):
        for epoch in range(self.num_epochs):
            for data_set in self.x[1:]:
                y_arr = [sum([neuron[i] * data_set[i] for i in range(len(data_set))]) for neuron in self.w]
                flag = sum(y_arr) != 0
                while flag:
                    j = y_arr.index(max(y_arr))
                    if y_arr[j] == 0:
                        break
                    r = sum([data_set[i] * self.t[j][i] for i in range(len(data_set))]) / sum(data_set)
                    if r > self.r_crit:
                        for i in range(len(self.w[j])):
                            self.w[j][i] = (1 - self.learning_rate) * self.w[j][i] + self.learning_rate * (self.LAMBDA * data_set[i]) / (self.LAMBDA - 1 + sum(self.x[0]))
                            self.t[j][i] = (1 - self.learning_rate) * self.t[j][i] + self.learning_rate * data_set[i]
                        flag = False
                    else:
                        y_arr[j] = 0
                if max(y_arr) == 0:
                    self.w.append([(self.LAMBDA * data_set[i]) / (self.LAMBDA - 1 + sum(data_set)) for i in range(len(data_set))])
                    self.t.append([data_set[i] for i in range(len(data_set))])
            self.learning_rate -= self.learning_rate_decrease
        return self.w
    
    def get_clusters(self):
        clusters = []
        for data_set in self.x:
            y_arr = [sum([neuron[i] * data_set[i] for i in range(len(data_set))]) for neuron in self.w]
            clusters.append(y_arr.index(max(y_arr)))
        return clusters


def main():
    x = [[1,0,1,0,1,0,1,0,1],
        [0,1,0,1,0,1,0,1,0],
        [1,0,1,0,1,0,1,0,0],
        [0,1,1,1,0,1,0,1,0]]
    
    nn = ART_1(x)
    
    nn.train()
    clusters = nn.get_clusters()
    print(clusters)


if __name__ == '__main__':
    main()