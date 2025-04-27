

import random
import math

class RNN_Net:
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        
        self.Wxh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(vocab_size)]  # x ----------> input
        self.Whh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]  # h ---------> hidden
        self.Why = [[random.uniform(-0.1, 0.1) for _ in range(vocab_size)] for _ in range(hidden_size)]   # y --------> output
        self.bh = [0.0 for _ in range(hidden_size)]  
        self.by = [0.0 for _ in range(vocab_size)]   

      
        self.inputs = None
        self.hidden_states = None
        self.outputs = None
    
    def tanh(self, x):     
        return math.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - x**2
    
    def softmax(self, x):  
        exp_x = [math.exp(i) for i in x]
        sum_exp_x = sum(exp_x)
        return [i/sum_exp_x for i in exp_x]
   


    def forward(self, inputs):   # Forward pass
        h_prev = [0.0 for _ in range(self.hidden_size)]  # Initialize previous hidden state with zeros
        self.inputs = inputs
        self.hidden_states = [h_prev.copy()]
        self.outputs = []
        
        
        for x in inputs:
            x_onehot = [0.0] * self.vocab_size
            x_onehot[x] = 1.0
            
            
            
            h = [0.0 for _ in range(self.hidden_size)] #  new hidden state
            for i in range(self.hidden_size):
                for j in range(self.vocab_size):
                    h[i] += self.Wxh[j][i] * x_onehot[j]
                for j in range(self.hidden_size):
                    h[i] += self.Whh[j][i] * h_prev[j]
                h[i] = self.tanh(h[i] + self.bh[i])
                
                
            
            y = [0.0 for _ in range(self.vocab_size)]  #  output prediction
            for i in range(self.vocab_size):
                for j in range(self.hidden_size):
                    y[i] += self.Why[j][i] * h[j]
                y[i] += self.by[i]
            y = self.softmax(y)
            
            
            
            self.hidden_states.append(h.copy())
            self.outputs.append(y.copy())
            h_prev = h  # Current hidden state  will becomes previous for next step
        
        return self.outputs
    
    
    
    def backward(self, targets, learning_rate=0.01):    # back propagation
        dWxh = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.vocab_size)]
        dWhh = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]
        dWhy = [[0.0 for _ in range(self.vocab_size)] for _ in range(self.hidden_size)]
        dbh = [0.0 for _ in range(self.hidden_size)]
        dby = [0.0 for _ in range(self.vocab_size)]
        dh_next = [0.0 for _ in range(self.hidden_size)]
        
        
        
        for t in reversed(range(len(self.inputs))):
            h = self.hidden_states[t+1]
            h_prev = self.hidden_states[t]
            y = self.outputs[t]
            x = self.inputs[t]
            
            
            
            x_onehot = [0.0] * self.vocab_size
            x_onehot[x] = 1.0
            target_onehot = [0.0] * self.vocab_size
            target_onehot[targets[t]] = 1.0
            
            
            
            dy = [y[i] - target_onehot[i] for i in range(self.vocab_size)] # Calculate error at output layer
            for i in range(self.hidden_size):
                for j in range(self.vocab_size):
                    dWhy[i][j] += h[i] * dy[j]
            for i in range(self.vocab_size):
                dby[i] += dy[i]
            
            dh = [0.0 for _ in range(self.hidden_size)]
            for i in range(self.hidden_size):
                for j in range(self.vocab_size):
                    dh[i] += self.Why[i][j] * dy[j]
                dh[i] += dh_next[i]
            
            
            dh_raw = [dh[i] * self.tanh_derivative(h[i]) for i in range(self.hidden_size)]
            for i in range(self.vocab_size):
                for j in range(self.hidden_size):
                    dWxh[i][j] += x_onehot[i] * dh_raw[j]
            for i in range(self.hidden_size):
                dbh[i] += dh_raw[i]
            for i in range(self.hidden_size):
                for j in range(self.hidden_size):
                    dWhh[i][j] += h_prev[i] * dh_raw[j]
            
            dh_next = [0.0 for _ in range(self.hidden_size)]
            for i in range(self.hidden_size):
                for j in range(self.hidden_size):
                    dh_next[i] += self.Whh[i][j] * dh_raw[j]
                    
                    
        
        # Update parameters
        for i in range(self.vocab_size):
            for j in range(self.hidden_size):
                self.Wxh[i][j] -= learning_rate * dWxh[i][j]
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                self.Whh[i][j] -= learning_rate * dWhh[i][j]
        for i in range(self.hidden_size):
            for j in range(self.vocab_size):
                self.Why[i][j] -= learning_rate * dWhy[i][j]
        for i in range(self.hidden_size):
            self.bh[i] -= learning_rate * dbh[i]
        for i in range(self.vocab_size):
            self.by[i] -= learning_rate * dby[i]
            
            
    
    def train(self, training_data, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            for sequence, target in training_data:
                outputs = self.forward(sequence)
                loss = -math.log(outputs[-1][target])
                total_loss += loss
                targets = [0] * (len(sequence) - 1) + [target]
                self.backward(targets, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(training_data)}")
                
                

# I know you care
word_to_idx = {"I": 0, "know": 1, "you": 2, "care": 3}
idx_to_word = {v: k for k, v in word_to_idx.items()}

training_data = [
    ([0, 1, 2], 3)  # "I know you" → "care"
]



# Train 
rnn = RNN_Net(vocab_size=4, hidden_size=2)
rnn.train(training_data, epochs=100, learning_rate=0.1)



# Test 
test_sequence = [0, 1, 2]  # "I know you"
output = rnn.forward(test_sequence)
predicted_word = idx_to_word[output[-1].index(max(output[-1]))]
print("\nInput: 'I know you' -> Predicted next word: '{}'".format(predicted_word))













