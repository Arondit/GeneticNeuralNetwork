import numpy as np
import random as r
import pickle
import matplotlib.pylab as plt
import copy
def bin():
    return r.randint(0,1)
def parent(distr):
    rt = r.randint(0,10000)
    if rt<distr[0]: return 0
    for i in range(1,len(distr)):
        if distr[i-1]<rt<distr[i]:
            return i  
def best(list):
    b = 0
    for i in range(len(list)):
        if list[i]>list[b]: b = i
    return b
def sigmoid(x):
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self,*, count, weights=None, activation_function=sigmoid):
        if weights is None: 
            self.__weights = list()
            for i in range(count):
                self.__weights.append(r.random()*2-1)
        else: self.__weights = list(weights)
        self.__activation_function = activation_function
    def f(self,x):
        return self.__activation_function(x)
    @property
    def weights(self):
        return self.__weights
    def set_weights(self, weights):
        self.__weights = weights

class Layer:
    def __init__(self, *,count, neurons=None, next_layer_count=0,bias=None):
        if neurons is None:
            self.__neurons=list()
            for i in range(count):
                self.__neurons.append(Neuron(count=next_layer_count))
        #if bias is None: self.__bias=r.random()*0.2 - 0.1
        if bias is None: self.__bias = r.random()
        else: self.__bias = bias
        self.__matrix = None
    @property
    def neurons(self):
        return self.__neurons
    @property
    def matrix(self):
        if self.__matrix is None:
            self.__matrix = np.zeros((len(self.neurons),len(self.neurons[0].weights)))
            for i in range(len(self.__matrix)):
                self.__matrix[i] = self.neurons[i].weights
            self.__matrix = self.__matrix.transpose()
        return self.__matrix
    @property
    def bias(self):
        return self.__bias
    def set_bias(self,bias):
        self.__bias = bias
    def set_weights(self, weigths):
        for i in range(len(weigths)):
            self.neurons[i].set_weights(weigths[i])
        self.__matrix = None
            
class Network:
    def __init__(self,layers_count, bias=None):
        count = len(layers_count)
        self.__layers = list()
        if bias is None: bias = [r.random for i in range(count-1)]
        for i in range(count-1):
            self.__layers.append(Layer(count = layers_count[i],next_layer_count=layers_count[i+1],bias=bias[i]))
        self.__layers.append(Layer(count = layers_count[i+1],bias=0))
    @property
    def layers(self):
        return self.__layers
    @property
    def bias(self):
        return [l.bias for l in self.layers]
    def set_weights(self,weights):
        for i in range(len(weights)):
            self.layers[i].set_weights(weights[i])
    def set_bias(self, bias):
        for i in range(len(bias)):
             self.layers[i].set_bias(bias[i])
    def __str__(self):
        res=''
        for s in self.layers:
             res+= str(s.matrix) +'\n'
        return res
    def __call__(self,values):
        for i in range(len(self.layers)-1):
         #   print('iteration', i)
         #   print(values)
            values = [np.tanh(x) for x in values]
          #  print(values)
          #  print(self.layers[i].matrix)
            values = np.dot(self.layers[i].matrix,values)+self.layers[i].bias
           # print(values)
        values = [sigmoid(x) for x in values]
        return [abs(x) for x in values]
    def get_answer(self,values):
        values = list(values)
        values = self(values)
        b = 0
        for i in range(len(values)):
            if values[i]>values[b]: b = i
        return b
    def mistake(self,learning_data):
        full_mistake = 0.0
        for key in learning_data.keys():
            answer = self(key)
            mistake = 0.0
            for i in range(len(answer)):
                mistake += ((i==learning_data[key]) - answer[i])**2
            full_mistake += np.sqrt(mistake)
        return full_mistake/len(learning_data.keys())
        
def generate(layers_count,*,learning_data, population_count = 8, needed_accuracy = 0.2):
    variants = [Network(layers_count) for i in range(population_count)]
    e=0
    survival_coeffs = []
    accuracies=[]
    for e in range(3000):
        z=0
        if e%500 == 0:
            print('epoch ', e,accuracies)
            print(survival_coeffs)
            for n in variants:
                print('number',z)
                for key in learning_data.keys():
                    print(key,':',n(key))
                print([x.bias for x in n.layers])
                z+=1
        #print(accuracies)
        accuracies = [n.mistake(learning_data) for n in variants]
        sum = 0.0
        for m in range(population_count):
            if accuracies[m] <= needed_accuracy: return variants[m]
            sum += 1/accuracies[m]
        #print(sum)
        survival_coeffs = [int((1/m)/sum*10000) for m in accuracies]
        #print(survival_coeffs)
        distr = [0 for i in range(population_count)]
        distr[0] = survival_coeffs[0]
        for i in range(1, len(survival_coeffs)):
            distr[i] = distr[i-1] + survival_coeffs[i]
        parents = []
        for i in range(population_count):
            first = parent(distr)
            while first is None: 
                first = parent(distr)
            second = first
            while second == first:
                second = parent(distr)
                while second is None: 
                    second = parent(distr)
            parents.append((first,second))
        s = variants[best(survival_coeffs)]
        new_variants = []
        for i in range(population_count):
            first = [copy.deepcopy(l.matrix.transpose()) for l in variants[parents[i][0]].layers]
            second = [copy.deepcopy(l.matrix.transpose()) for l in variants[parents[i][1]].layers]
            both = (first,second)
            res = [[[both[bin()][i][j][k] for k in range(len(first[i][j]))] for j in range(len(first[i]))] for i in range(len(first))]
            for z in res:
                if z!= []:
                    for j in z:
                        if bin()*bin() == 1 and len(j)>0: j[r.randint(0,len(j)-1)] = r.random()
                    z = np.matrix(z)
                    z = z.transpose()
            new_variants.append(Network(layers_count,bias=[variants[parents[i][bin()]].bias[l] for l in range(len(variants[i].bias))]))
            new_variants[i].set_weights(res)
            
        variants = new_variants
        variants[0] = s
        #e+=1

learning_data = {(0,0):0, (0,1):1,(1,0):1,(1,1):0}

n = generate([2,3,4,2],learning_data=learning_data)
for key in learning_data.keys():
    print(key,':',n(key))


#x = np.arange(-8, 8, 0.1)
#f = 1 / (1 + np.exp(-x))

#plt.plot(x, f)
#plt.xlabel('x')
#plt.ylabel('f(x)')
#plt.show()


#FILENAME = "user.dat"
 
#with open(FILENAME, "wb") as file:
 #   pickle.dump(n, file)
 
#with open(FILENAME, "rb") as file:
 #   a = pickle.load(file)
aaaa
