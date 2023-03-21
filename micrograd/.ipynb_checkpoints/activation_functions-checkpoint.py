import math

class activations:
    def __init__(self,arr_data):
        self.arr_data = arr_data

    def tanh(self):
        x = self.arr_data
        y = []
        for i in x:
            # y.append(math.tanh(i))
            top = math.exp(1)**(2*i) - 1
            bot = math.exp(1)**(2*i) + 1
            y.append(top/bot)
        return(x,y)

    def relu(self):
        x = self.arr_data
        y = []
        for i in x:
            if(i<0):
                y.append(0)
            else:
                y.append(i)
        return(x,y)

    def sigmoid(self):
        x = self.arr_data
        y = []
        for i in x:
            y.append(1/(1+math.e**-i))

        return(x,y)

    def gaussian(self):
        x = self.arr_data
        y = []

        for i in x:
            y.append(math.e**-(i**2))

        return(x,y)

    def step(self):
        x = self.arr_data
        y = []

        for i in x:
            if(i < 0):
                y.append(0)
            else:
                y.append(1)
        return (x,y)

    def softplus(self):
        x = self.arr_data
        y = []

        for i in x:
            y.append(math.log(1 + math.e**i))

        return (x,y)

    def leaky_relu(self):
        x = self.arr_data
        y = []

        for i in x:
            if(i<0):
                y.append(0.1 * i)
            else:
                y.append(i)
        return(x,y)

