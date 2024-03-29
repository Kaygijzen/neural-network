class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        batch_size = len(input_data)
        result = []

        for i in range(batch_size):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        
        return result
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        batch_size = len(x_train)

        for i in range(epochs):
            err = 0

            for j in range(batch_size):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= batch_size
            print(f"Epoch {i + 1}/{epochs} error = {err}")


                
