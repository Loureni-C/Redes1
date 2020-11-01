import numpy as np


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((20, 10)) - 1
        self.synaptic_weights2 = 2 * np.random.random((10, 6)) - 1
        self.synaptic_weights3 = 2 * np.random.random((6, 6)) - 1
        self.synaptic_weights4 = 2 * np.random.random((6, 1)) - 1
        

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def saida(self, inputs):
        inputs = inputs.astype(float)
        
        self.lay0 = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        self.lay1 = self.sigmoid(np.dot(self.lay0, self.synaptic_weights2))
        self.lay2 = self.sigmoid(np.dot(self.lay1, self.synaptic_weights3))
        self.output = self.sigmoid(np.dot(self.lay2, self.synaptic_weights4))
        output0= self.output

        return output0

    def treino(self, training_inputs, training_outputs,  training_iterations):

        for iteration in range(training_iterations):

            
            output = self.saida(training_inputs)
            error = training_outputs - output
            error_q = ((training_outputs - output)**2)/2
            
            self.synaptic_weights4 += np.dot(self.lay2.T, 0.01*error_q*self.sigmoid_derivative(output))

            error2 = self.synaptic_weights4*error*self.sigmoid_derivative(self.lay2)
            self.synaptic_weights3 += np.dot(self.lay1.T, 0.01*error2)
            
##            print('blaaaaaaaaaaaaaaa\n',self.sigmoid_derivative(self.lay1))
##            
##            print('oiiiiiiiii\n',self.synaptic_weights4)
            
            error3 = self.synaptic_weights3*error2*self.sigmoid_derivative(self.lay1)
            self.synaptic_weights2 += np.dot(self.lay0.T, 0.01*error3)

            
##            print('6X6\n',self.synaptic_weights3)
##            print('6X6\n',error2)
##            print('6X6\n',self.sigmoid_derivative(self.lay1))  
##            print('6X10 lay 0 normal\n',self.lay0)
##            print('lay3 transposto\n',self.lay0.T)
##            print('erro 3\n',error3)

            
##            print('pesos 20X10\n',self.synaptic_weights2)
##            print('derivada lay 0\n',self.sigmoid_derivative(self.lay0))
##            print('erro 3\n',error3)

            error4 = np.dot(self.synaptic_weights2*self.sigmoid_derivative(self.lay0.T),error3)
            self.synaptic_weights += np.dot(training_inputs.T, 0.01*error4.T)








            
if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("pesos entrada")
    print(neural_network.synaptic_weights,'\n')
    print("pesos layer1")
    print(neural_network.synaptic_weights2,'\n')
    print("pesos layer2")
    print(neural_network.synaptic_weights3,'\n')
    print("pesos layer3")
    print(neural_network.synaptic_weights4,'\n')

    training_inputs = np.array([[0,0,1,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,1,1],          
                                [1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1],            
                                [1,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0],            
                                [0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1],            
                                [0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1],            
                                [1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1]])         

    training_outputs = np.array([[0,0,0,1,1,1]]).T

    training_iterations = 50000

    neural_network.treino(training_inputs, training_outputs, training_iterations)

    print("pesos treinados após o treinamento layer0")
    print(neural_network.synaptic_weights,'\n')
    print("pesos treinados após o treinamento layer1")
    print(neural_network.synaptic_weights2,'\n')
    print("pesos treinados após o treinamento layer2")
    print(neural_network.synaptic_weights3,'\n')
    print("pesos treinados após o treinamento layer3")
    print(neural_network.synaptic_weights4,'\n')



    A = str(input("entrada 1: "))
    B = str(input("entrada 2: "))
    C = str(input("entrada 3: "))
    D = str(input("entrada 4: "))
    E = str(input("entrada 5: "))
    F = str(input("entrada 6: "))
    G = str(input("entrada 7: "))
    H = str(input("entrada 8: "))
    I = str(input("entrada 9: "))
    J = str(input("entrada 10: "))
    K = str(input("entrada 11: "))
    L = str(input("entrada 12: "))
    M = str(input("entrada 13: "))
    N = str(input("entrada 14: "))
    O = str(input("entrada 15: "))
    P = str(input("entrada 16: "))
    Q = str(input("entrada 17: "))
    R = str(input("entrada 18: "))
    S = str(input("entrada 19: "))
    T = str(input("entrada 20: "))


    print("Nova entrada eh: ", A, B, C, D, E, F, G, H,I,J,K,L,M,N,O,P,Q,R,S,T)
    print("classificacao da rede: ")
    print(neural_network.saida(np.array([A,B,C,D,E,F,G, H,I,J,K,L,M,N,O,P,Q,R,S,T])))
