from matplotlib import pyplot as plt


class ClusteredData:
    def __init__(self, labels=None, data=None):
        self.labels, self.data = labels, data
        self.clustered = dict()
        self.non_clustered = []

    def compose(self):
        self.clear()

        if len(self.labels) != len(self.data):
            raise Exception("Size of labels is not the same as size of data: "
                            + str(len(self.labels)) + " and " + str(len(self.data)))

        for i in range(len(self.labels)):
            if self.labels[i] == -1:
                self.non_clustered.append(self.data[i])
            elif self.labels[i] in self.clustered:
                self.clustered[self.labels[i]].append(self.data[i])
            else:
                self.clustered[self.labels[i]] = [self.data[i]]

    def fill(self, clustered_data, non_clustered=None):
        self.clear()
        self.clustered = clustered_data
        self.non_clustered = non_clustered

    def get_num(self):
        return len(set(self.labels)) - (1 if -1 in self.labels else 0)

    def boundaries(self):
        bounds = []
        for key in self.clustered.keys():
            times = [row[0] for row in self.clustered[key]]
            bounds.append((min(times), max(times)))
        return bounds

    def clear(self):
        self.clustered = dict()
        self.non_clustered = []

    def show(self):
        plt.scatter(self.data[:, 0], self.data[:, 0], c=self.labels, cmap='viridis', s=50)
        plt.title('DBSCAN Clustering')
        plt.show()

    def __str__(self):
        print_str = ""

        if len(self.clustered) == 0:
            print_str = "Cluster is empty."
        else:
            for key, values in self.clustered.items():
                print_str += "Cluster " + str(key) + ":" + "\n"
                for val in values:
                    print_str += str(val) + "\n"

        return print_str
