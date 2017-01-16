from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
import copy

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        # yt[yt == 0] = -1
        yt[yt == 1] = .99
        yt[yt == 0] = .01
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        inv = np.linalg.pinv(Xtrain)/numsamples
        # self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        self.weights = np.dot(inv,pow((np.subtract((pow((2*yt-1), -2)),1)),-.5))*-1/numsamples

    def predict(self, Xtest):
        # ytest = np.dot(Xtest, self.weights)
        ytest = self._func(Xtest)
        ytest[ytest > 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest

    def _func(self, X):
        t = np.add(1,np.divide(np.dot(X, self.weights.T),pow(np.add(1,pow(np.dot(X, self.weights.T),2)),.5)))*.5
        return t
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'usecolumnones': False}
        self.reset(parameters)
            
    def reset(self, parameters):
        self.resetparams(parameters)
        self.mu_1 = None
        self.sigma_1 = None
        self.mu_0 = None
        self.sigma_0 = None
        self.y_prob = None
        # TODO: set up required variables for learning

    def learn(self, Xtrain, ytrain):
        if self.params['usecolumnones'] == False:
            Xtrain1 = Xtrain[:,0:len(Xtrain[0])-1]
        else:
            Xtrain1 = Xtrain
        c1 = 0
        c0 = 0
        y_p = []
        for u in range(len(ytrain)):
            if ytrain[u] == 1:
                c1 += 1
            else:
                c0 += 1
        mydict = {}
        mydict[1] = c1
        mydict[0] = c0
        mu_sum_1 = np.zeros(len(Xtrain1[0]))
        mu_sum_0 = np.zeros(len(Xtrain1[0]))
        sigma_sum_1 = np.zeros(len(Xtrain1[0]))
        sigma_sum_0 = np.zeros(len(Xtrain1[0]))
        for i in range(len(Xtrain1[0])):
            for j in range(len(Xtrain1)):
                if ytrain[j] == 1:
                    mu_sum_1[i] += Xtrain1[j][i]
                else:
                    mu_sum_0[i] += Xtrain1[j][i]
        mu_sum_1 = mu_sum_1/mydict[1]
        mu_sum_0 = mu_sum_0/mydict[0]
        for i in range(len(Xtrain1[0])):
            for j in range(len(Xtrain1)):
                if ytrain[j] == 1:
                    sigma_sum_1[i] += pow((Xtrain1[j][i] - mu_sum_1[i]),2)
                else:
                    sigma_sum_0[i] += pow((Xtrain1[j][i] - mu_sum_0[i]),2)
        for x in range(len(sigma_sum_1)):
            sigma_sum_1[x] = math.sqrt(sigma_sum_1[x]/(mydict[1]-1))
            sigma_sum_0[x] = math.sqrt(sigma_sum_0[x]/(mydict[0]-1))
        self.mu_0 = mu_sum_0
        self.mu_1 = mu_sum_1
        self.sigma_0 = sigma_sum_0
        self.sigma_1 = sigma_sum_1
        y_p.append(mydict[1]/(mydict[1]+mydict[0]*1.0))
        y_p.append(mydict[0]/(mydict[1]+mydict[0]*1.0))
        self.y_prob = y_p

    def predict(self, Xtest):
        ytest = []
        if self.params['usecolumnones'] == False:
            Xtest1 = Xtest[:,0:len(Xtest[0])-1]
        else:
            Xtest1 = Xtest
        for i in range(len(Xtest1)):
            prob_1 = 1
            prob_0 = 1
            for j in range(len(Xtest1[0])):
                prob_1 *= (utils.calculateprob(Xtest1[i][j], self.mu_1[j], self.sigma_1[j]))
                prob_0 *= (utils.calculateprob(Xtest1[i][j], self.mu_0[j], self.sigma_0[j]))
            if prob_1*self.y_prob[0] > prob_0*self.y_prob[1]:
                ytest.append(1)
            else:
                ytest.append(0)
        return ytest



    # TODO: implement learn and predict functions                  
            
class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        numsamples = Xtrain.shape[0]
        I = np.identity(Xtrain.shape[0])
        Id = np.identity(Xtrain.shape[1])
        # self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),yt)
        self.weights = np.zeros(len(Xtrain[0]))
        param = .02
        reg = Id
        if self.params['regularizer'] == 'None':
            reg = 0
            reg1 = 0*Id
        if self.params['regularizer'] == 'l1':
            reg = self.params['regwgt']
            reg1 = 0*Id

        for i in range(30):
            if self.params['regularizer'] == 'l2':
                reg = self.params['regwgt']*self.weights
                reg1 = self.params['regwgt']*Id
            pw = utils.sigmoid(np.dot(Xtrain, self.weights.T))
            Pw = pw*np.identity(Xtrain.shape[0])
            IP = np.subtract(I, Pw)
            yp = np.subtract(yt, pw)
            fac = np.dot(np.linalg.pinv(np.subtract(np.dot(np.dot(Xtrain.T, Pw), np.dot(IP, Xtrain)),reg1)),np.subtract(np.dot(Xtrain.T, yp),reg))
            self.weights = np.add(self.weights, fac)
            # self.weights = self.weights + param*np.dot(Xtrain.T, yp)
            # param = param/pow(i+1, .5)



    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights.T))
        ytest[ytest > 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest
     
    # TODO: implement learn and predict functions                  
           

class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                       'transfer': 'sigmoid',
                       'stepsize': 0.01,
                       'epochs': 100}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
            self.ni = 5
            self.no = 1

        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
            self.wi = None
            self.wo = None

        node_number=copy.deepcopy(self.params['nh'])
        node_input=copy.deepcopy(self.ni)
        node_output=copy.deepcopy(self.no)
        self.wi = 4*np.random.random_sample((node_number, node_input))-3
        self.wo = 4*np.random.random_sample((node_output,node_number))-3

    def learn(self, Xtrain, ytrain):
        for echo in range(self.params['epochs']):
            for t in range(Xtrain.shape[0]):
                self.calc(Xtrain[t], ytrain[t])
    
    def calc(self, Xtrain, ytrain):
        ah, ao = self.evaluate(Xtrain)
        Xtrain = np.reshape(Xtrain.T, (1, Xtrain.shape[0]))
        a = (-(ytrain / ao) + (1 - ytrain) / (1 - ao))
        b = (ao * (1 - ao))
        c = a*b
        value = np.dot(self.wi, Xtrain.T)
        t_value = self.transfer(value)
        delta1 = c * t_value.T
        d_value = self.dtransfer(value)
        delta2 = c * np.multiply(self.wo.T, d_value)
        self.wo = self.wo - self.params['stepsize'] * delta1
        self.wi = self.wi - self.params['stepsize'] * delta2

    def evaluate(self, inputs):
        """
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """

        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')

        # hidden activations
        ah = self.transfer(np.dot(self.wi, inputs))

        # output activations
        ao = self.transfer(np.dot(self.wo, ah))
        return (ah, ao)


    def predict(self, Xtest):
        ytest=[]
        value = self.transfer(np.dot(self.wi, Xtest.T))
        value=np.insert(value, -1, 1, axis=1)
        predict = self.transfer(np.dot(self.wo, value))
        for s in predict[0]:
            if s<0.5:
                ytest.append([0])
            else:
                ytest.append([1])
        return ytest



class LogitRegAlternative(Classifier):

    def __init__( self, parameters={} ):
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        numsamples = Xtrain.shape[0]
        I = np.identity(Xtrain.shape[0])
        Id = np.identity(Xtrain.shape[1])
        # self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),yt)
        self.weights = np.zeros(len(Xtrain[0]))

        for i in range(50):
            reg = .0003*self.weights + .03
            reg1 = .0003*Id
            # reg = .1/(math.sqrt(1+.05))*self.weights*Id
            pw = utils.sigmoid(np.dot(Xtrain, self.weights.T))
            Pw = pw*np.identity(Xtrain.shape[0])
            IP = np.subtract(I, Pw)
            yp = np.subtract(yt, pw)
            fac = np.dot(np.linalg.pinv(np.subtract(np.dot(np.dot(Xtrain.T, Pw), np.dot(IP, Xtrain)),reg1)),np.subtract(np.dot(Xtrain.T, yp),reg))
            self.weights = np.add(self.weights, fac)
            # self.weights = self.weights + param*np.dot(Xtrain.T, yp)
            # param = param/pow(i+1, .5)

    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights.T))
        ytest[ytest > 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest
        
    # TODO: implement learn and predict functions

class Radial_Basis_Func(Classifier):
    def __init__(self, parameters={}):
        self.params = parameters
        self.k = self.params['k']
        self.sigma = self.params['s']
        self.centers = []

    def reset(self, parameters):
        self.resetparams(parameters)
        if "regularizer" in self.params:
            if self.params['regularizer'] is 'l1':
                self.regularizer = (utils.l1, utils.dl1)
            elif self.params['regularizer'] is 'l2':
                self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def resetparams(self, parameters):
        self.params['regwgt'] = parameters['regwgt']

    def learn(self, Xtrain, ytrain):
        step_size=0.001      
        epsilon = 0.0001 
        convergence=False
        count = 0
        self.centers = []
        a = np.arange(Xtrain.shape[0])
        np.random.shuffle(a)
        local = a[:self.k]
        for i in local:
            self.centers.append(Xtrain[i])
        self.centers = np.array(self.centers)
        phi = []
        for x in (Xtrain):
            trans = []
            for c in (self.centers):
                dist = np.linalg.norm(x-c)*np.linalg.norm(x-c)
                value =math.exp (-self.sigma * dist)
                trans.append(value)
            phi.append(trans)
        phi = np.array(phi)
        Xtrain1=phi
        self.weights = np.zeros(Xtrain1.shape[1])

        while (convergence == False):
            oldweights = copy.deepcopy(self.weights)
            score = np.dot(Xtrain1, self.weights)
            pred = utils.sigmoid(score)
            derivative = np.dot(np.transpose(Xtrain1), np.subtract(ytrain, pred))
            self.weights += step_size * derivative
            newweights = copy.deepcopy(self.weights)
            count += 1
            diff = np.subtract(newweights, oldweights)
            val = np.sum(np.power(diff, 2))
            if (np.sqrt(val)) < epsilon:
                convergence = True
        return self.weights



    def predict(self, Xtest):
        lab = []
        for x in (Xtest):
            newrow=[]
            for c in (self.centers):
                dist = np.linalg.norm(x-c)*np.linalg.norm(x-c)
                value = math.exp(-self.sigma* dist)
                newrow.append(value)
            lab.append(newrow)

        lab=np.array(lab)
        scores = np.dot(lab, self.weights)
        predicts = utils.sigmoid(scores)
        threshold_p = utils.threshold_probs(predicts)
        return threshold_p
           
    
