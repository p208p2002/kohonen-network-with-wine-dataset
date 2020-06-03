import logging
import SimpSOM as sps
import numpy as np

FORMAT = '%(filename)s line:%(lineno)d\t%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT)
print = logging.info

def make_XY(data):
    X,Y = [],[]
    for d in data:
        d = d.split(',')
        X.append(d[1:])
        Y.append(d[0])
    #
    Y = [int(y) for y in Y]
    X = np.array([np.array([float(f) for f in x]) for x in X])
                
    return X,Y
    
if __name__ == "__main__":
    with open('wine.data','r',encoding='utf-8') as f:
        data = f.read()
    data = data.split('\n')[:-1]
    X,Y = make_XY(data)

    FEATURE_COUNT = len(X[0])
    CLASS_COUNT = len(list(set(Y)))
    print('feature_count:%d'%FEATURE_COUNT)
    print('class_count:%d'%CLASS_COUNT)

    net = sps.somNet(20, 20, X, PBC=True)
    #Train the network for 10000 epochs and with initial learning rate of 0.01. 
    net.train(0.008, 15000)

    #Save the weights to file
    net.save('filename_weights')

    #Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
    #and then according to the distance between each node and its neighbours.
    net.nodes_graph(colnum=0)
    net.diff_graph()

    #Project the datapoints on the new 2D network map.
    net.project(X, labels=Y)

    #Cluster the datapoints according to the Quality Threshold algorithm.
    net.cluster(X, type='qthresh')	
