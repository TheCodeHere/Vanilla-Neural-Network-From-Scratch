from mnist.loader import MNIST
import numpy as np
import sklearn.metrics as skmet
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import decomposition

def GetData():
    mndata = MNIST(gz=True)

    Data_train, labels_train = mndata.load_training()
    Data_test, labels_test = mndata.load_testing()

    classes = 10

    print("Adequating data ", end="", flush=True)

    # Making sure that the values are float so that we can get decimal points after division
    Data_train = np.array(Data_train, dtype=float)
    Data_test = np.array(Data_test, dtype=float)

    # Normalizing the RGB codes
    Data_train /= 255
    Data_test /= 255

    #Labels to one-hot format
    labels_train = np.eye(classes)[labels_train]
    labels_test = np.eye(classes)[labels_test]

    print("...Done.")

    return Data_train, labels_train, Data_test, labels_test

def Inicialization(NpL):
    #Initialize weights values randomly [0,0.01) and bias values [1's]
    W1 = np.random.rand(NpL[1], NpL[0]) * 0.01 #np.sqrt(2.0/NpL[0])
    b1 = np.ones((NpL[1], 1))

    W2 = np.random.rand(NpL[2], NpL[1]) * 0.01 #np.sqrt(2.0/NpL[1])
    b2 = np.ones((NpL[2], 1))

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
'''
def Ploting(tit = "default", X='X1', Y='X2'):
    plt.grid(True,linestyle='--')#add grid

    plt.title(tit) #add title

    # add x,y axes labels
    plt.xlabel(X)
    plt.ylabel(Y)
'''
def Evaluation(testX, testY, params):
    #One-by-One
    '''
    pred_class = []
    true_class = []
    for x,y in zip(testX,testY):
        pred_class.append(np.argmax(ForwardProp(x.reshape(-1, 1), y.reshape(-1, 1), params)['yh']))
        true_class.append(np.argmax(y.reshape(-1, 1)))
    '''
    # All-at-once
    pred_class = np.argmax(ForwardProp(testX.T, testY.T, params)['yh'], axis=0) #(1,n)
    true_class = np.argmax(testY.T, axis=0)

    print("The average scores for all classes:")
    # Calculate metrics for each label, and find their unweighted mean. does not take label imbalance into account.
    print("\nAccuracy:  {:.2f}%".format(skmet.accuracy_score(true_class, pred_class)*100)) #(TP+TN)/Total / classes
    print("Precision: {:.2f}%".format(skmet.precision_score(true_class, pred_class, average='macro')*100)) #TP/(TP+FP) / classes
    print("Recall:    {:.2f}%".format(skmet.recall_score(true_class, pred_class, average='macro')*100)) #TP/(TP+FN) / classes
    print("F-measure: {:.2f}%".format(skmet.f1_score(true_class, pred_class, average='macro')*100)) #2 * (prec*rec)/(prec+rec) / classes

    print("\nThe scores for each class:")
    precision, recall, fscore, support = skmet.precision_recall_fscore_support(true_class, pred_class)

    print("\n| Label | Precision | Recall | F1-Score | Support")
    print("|-------|-----------|--------|----------|---------")
    for i in range(len(precision)):
        print("|   {}   |  {:.2f}%   | {:.2f}% |   {:.2f}   |   {}".format(i,precision[i]*100,recall[i]*100,fscore[i],support[i]))

'''
def cross_entropy_loss(yh, y, W1, W2, m, lamb):
    yh_log = np.log2(yh, out=np.zeros_like(yh), where=(yh != 0))# Suppress runtimewarning divide by zero encountered in log
    loss = -1.0/m * np.dot(y.T,yh_log)

    ## L2-Regularization ###
    loss += (lamb/(2*m)) * (np.linalg.norm(W1, 'fro')**2 + np.linalg.norm(W2, 'fro')**2)

    cost.append(loss.item())
'''

def cross_entropy_loss(Yh, Y, W1, W2, n, lamb):
    Yh_log = np.log2(Yh, out=np.zeros_like(Yh), where=(Yh != 0))# Suppress runtimewarning divide by zero encountered in log
    loss = np.multiply(Y.T,Yh_log)
    loss = -1.0 * np.sum(loss)

    ## L2-Regularization ###
    loss += (lamb/2.0) * (np.linalg.norm(W1, 'fro')**2 + np.linalg.norm(W2, 'fro')**2)

    return loss / n


def Softmax(x):
    '''Squashes a vector in the range (0, 1) and all the resulting elements add up to 1,
        they can be interpreted as class probabilities.'''
    m = np.max(x,axis=0)
    exp = np.exp(x-m)
    sum_exp = np.sum(exp,axis=0)

    return exp/sum_exp
'''
def Softmax(x):
    #Squashes a vector in the range (0, 1) and all the resulting elements add up to 1,
    #they can be interpreted as class probabilities.
    m = np.max(x)
    exp = np.exp(x-m)
    sum_exp = np.sum(exp)

    return exp/sum_exp
'''
def Relu(x):
    return np.maximum(x, 0)

def ForwardProp(x, y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]

    # Hidden Layer 1
    z1 = np.dot(W1,x) + b1 #(20,784)(784,1)+(20,1)
    a1 = Relu(z1) #(20,1)

    #Output layer
    z2 = np.dot(W2, a1) + b2 #(10,20)(20, 1)+(10,1)
    yh = Softmax(z2) #(10,1)

    return {'x': x, 'y': y, 'a1': a1, 'yh': yh}

def BackProp(f_prop, params, lamb):
    W1, W2 = [params[key] for key in ('W1', 'W2')]
    x, y, a1, yh = [f_prop[key] for key in ('x', 'y', 'a1', 'yh')]

    # Output Layer
    dz2 = (yh - y) / yh.shape[0]  #(10,1)-(10,1)

    dW2 = np.dot(dz2, a1.T)  #(10,1)(1,20)
    db2 = dz2

    # Hidden Layer 1
    dz1 = np.dot(W2.T, dz2) #(20,10)(10,1)
    dz1[a1<=0] = 0 #(20,1) [RELU derivation] if z1 > 0 -> a1 > 0, if z1 <= 0 -> a1 = 0

    dW1 = np.dot(dz1, x.T) #(20,1)(1,784)
    db1 = dz1

    ### L2-Regularization ###
    dW2 += lamb * W2 #(10,20) += (beta)(10,20)
    dW1 += lamb * W1 #(20,784) += (beta)(20,784)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

def Grad_desc(params, update, alfa):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    dW1, db1, dW2, db2 = [update[key] for key in ('dW1', 'db1', 'dW2', 'db2')]

    W1 -= alfa * dW1 #(20,784) -= (alfa)*(20,784)
    b1 -= alfa * db1 #(20,1) -= (alfa)*(20,1)

    W2 -= alfa * dW2 #(20,10) -= (alfa)*(20,10)
    b2 -= alfa * db2 #(10,1) -= (alfa)*(10,1)

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def Train(X, Y, NpL, Epochs, alfa = 0.01, lamb = 0.0001, plot_cost = True):
    Parameters = Inicialization(NpL) #initialize Weights & bias

    cost = []
    n = X.shape[0]

    ''' Stochastic Gradient Descent '''
    for epoch in range(1,Epochs+1):
        print("### Epoch ", epoch, " / ", Epochs, " ###")
        for x, y in zip(X, Y):
            f_prop = ForwardProp(x.reshape(-1, 1), y.reshape(-1, 1), Parameters)
            b_prop = BackProp(f_prop, Parameters, lamb)
            Parameters = Grad_desc(Parameters, b_prop, alfa)
        if plot_cost:
            Yh = ForwardProp(X.T, Y.T, Parameters)['yh']
            cost.append(cross_entropy_loss(Yh, Y, Parameters['W1'], Parameters['W2'], n, lamb))

    ### plot the cost function ###
    if plot_cost:
        plt.figure()
        plt.plot(np.arange(1,Epochs+1),cost, '-b')

        plt.grid(True, linestyle='--')  # add grid
        plt.title("Cost Function - Vanilla NN")  # add title

        # add x,y axes labels
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
    ###############################

    return Parameters

def Data_Visualization(testX, testY, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]

    # Forward Propagation without Softmax step
    z1 = np.dot(W1, testX.T) + b1
    a1 = Relu(z1)
    z2 = (np.dot(W2, a1) + b2).T

    #convert one-hot labels to int values
    true_class = np.argmax(testY, axis=1)

    #PCA for dimensionality reduction of the data (3D)
    pca = decomposition.PCA(n_components=3)
    pca.fit(z2)
    X_pca = pca.transform(z2)

    ######### Plot Data #########
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    scatter = ax.scatter(X_pca.T[0], X_pca.T[1], X_pca.T[2], s=5, c=true_class)

    legend_classes = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend_classes)

    plt.title("Data Visualization")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ##############################

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = GetData()

    n_train, m_train = X_train.shape  #Samples(rows),features(columns)
    print("X_train size: ", n_train, ",", m_train)
    print("Y_train size: ", Y_train.shape)

    n_test, m_test = X_test.shape
    print("X_test size: ", n_test, ",", m_test)
    print("Y_test size: ", Y_test.shape)

    NpL = [m_train,20,10] #Neurons per layer

    print("\nNeurons per layer: ")
    print("Layer 1: ", NpL[0])
    print("Layer 2: ", NpL[1])
    print("Layer 3: ", NpL[2],"\n")

    print("##############################################################\n")
    print("Training... \n")
    Parameters = Train(X_train, Y_train, NpL, 7, 0.09, 0.00009) #(30, 0.09, 0.00009)
    print("\n...Done.\n")

    print("Evaluating... \n")
    Evaluation(X_test, Y_test, Parameters)
    print("\n...Done.\n")

    print("Plotting Data... ",end="")
    Data_Visualization(X_test, Y_test, Parameters)
    print("Done.\n")

    print("\n... Finished.\n")

    plt.show()
