class Dataset_Loader():
    def __init__(self, filename):
        self.name=filename
    def load_monk1(self):
        self.x = []
        self.y = []
        f = open(self.name, 'r')
        for line in f:
            line = line.strip()
            elements = line.split()

            self.y.append([float(elements[0])]) 
            print elements[0]
            elements = elements[1:-1]
            feature = [[0 for i in range(0, n_classes)] for n_classes in [3, 3, 2, 3, 4, 2]]
            for i, n_classes in enumerate([3, 3, 2, 3, 4, 2]):
                feature[i][int(elements[i])-1] = 1
            final_features = []
            for v in feature:
                for el in v:
                    final_features.append(el)
            self.x.append(final_features)
        
        f.close()
