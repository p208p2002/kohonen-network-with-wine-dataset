import numpy as np
from som import SOM
import logging


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
    Y = [int(y)-1 for y in Y]
    X = np.array([np.array([float(f) for f in x]) for x in X])
                
    return X,Y
    

if __name__ == "__main__":
    with open('wine.data','r',encoding='utf-8') as f:
        data = f.read()
    data = data.split('\n')[:-1]
    X,Y = make_XY(data)

    som = SOM(8, 8)  # initialize the SOM
    som.fit(X, 10000, save_e=True, interval=100)  # fit the SOM for 10000 epochs, save the error every 100 steps
    som.plot_error_history(filename='images/som_error.png')  # plot the training error history

    # now visualize the learned representation with the class labels
    som.plot_point_map(X, Y, ['Class %d'%(l+1) for l in range(3)], filename='images/som.png')
    som.plot_class_density(X, Y, t=0, name='Class 0', filename='images/class_0.png')
    som.plot_distance_map(filename='images/distance_map.png')  # plot the distance map after training