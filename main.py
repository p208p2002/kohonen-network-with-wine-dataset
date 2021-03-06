import numpy as np
from som import SOM
import logging


FORMAT = '%(filename)s line:%(lineno)d\t%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT)
print = logging.info

def feature_normalization(X):
    count_featrue = X.shape[-1]
    f_maxs = []
    f_mins = []

    for i in range(count_featrue):
        f_max = np.max(X[:,[i]])
        f_min = np.min(X[:,[i]])
        f_maxs.append(f_max)
        f_mins.append(f_min)

    for i in range(len(X)):
        x_normalizatoion = np.array([0.0]*13)
        for j in range(len(X[i])):
            f_max = f_maxs[j]
            f_min = f_mins[j]
            x_normalizatoion[j] = (X[i][j] - f_min)/(f_max - f_min)
        X[i] = x_normalizatoion
    return X

def make_XY(data):
    X,Y = [],[]
    for d in data:
        d = d.split(',')
        X.append(d[1:])
        Y.append(d[0])
    #
    Y = [int(y)-1 for y in Y]
    X = np.array([np.array([float(f) for f in x]) for x in X])      
    return X,Y
    
if __name__ == "__main__":
    with open('wine.data','r',encoding='utf-8') as f:
        data = f.read()
    data = data.split('\n')[:-1]
    X,Y = make_XY(data)
    X = feature_normalization(X)
    # exit()
    FEATURE_COUNT = len(X[0])
    CLASS_COUNT = len(list(set(Y)))
    print('FEATURE_COUNT:%d'%FEATURE_COUNT)
    print('CLASS_COUNT:%d'%CLASS_COUNT)

    som = SOM(8, 8)  # initialize the SOM
    som.fit(X, 10000, save_e=True, interval=100)  # fit the SOM for 10000 epochs, save the error every 100 steps
    som.plot_error_history(filename='images/som_error.png')  # plot the training error history

    # now visualize the learned representation with the class labels
    som.plot_point_map(X, Y, ['Class %d'%(l+1) for l in range(CLASS_COUNT)], filename='images/som.png')
    for i in range(CLASS_COUNT):        
        som.plot_class_density(X, Y, t=i, name='Class %d'%(i+1), filename='images/class_%d.png'%(i+1))
    som.plot_distance_map(filename='images/distance_map.png')  # plot the distance map after training