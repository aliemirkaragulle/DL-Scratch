from sklearn import datasets # type: ignore
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# NN w/ No Hidden Layer
class Perceptron:
    def __init__(self, X, y):
        # X and y Should be NumPy Arrays for this Class
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.X = X

        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self.y = y.reshape(-1,1) # Make y a Column Vector

        # Shape of X and y
        self.X_shape = X.shape
        self.y_shape = self.y.shape
        self.m, self.n = X.shape

        # Initialize the Weights (A Weight for each Feature in X) and the Bias 
        self.w = np.zeros((self.n, 1))
        self.b = 0.0
    
    def linear_combination(self):
        # z = w.T*X + b
        # A Linear Combination, wT.X + b for Each Observation (Rows)
        z = np.zeros((self.m, 1))

        # For Each Feature Vector
        for i in range(self.m):
            # Get the Feature Vector and Calculate wT.X
            fv = self.X[i,:]
            z[i] = np.dot(fv, self.w) # Dimensions = (1, n) x (n, 1)
        # w.T*X + b
        z += self.b
        z = z.reshape(-1,1) # Reshape z into a Column Vector to Match with the Dimensions of y (m, 1)

        return z 

    def linear_combination_v(self):
        z = np.dot(self.X, self.w) + self.b # Dimensions: (m, n) x (n, 1) + R = (m, 1)
        z = z.reshape(-1,1) # Reshape z into a Column Vector to Match with the Dimensions of y
        return z
    
    def sigmoid(self, z):
        z = np.clip(z, -1000, 1000) 
        return 1 / (1 + np.exp(-z)) # Dimensions = (m, 1)
       
    def J(self, a):
        # Binary Cross-Entropy Loss/ Log-Loss
        # a is the Output of the Sigmoid Function
        a = np.clip(a, 1e-10, 1 - 1e-10)
        logloss = -((self.y * np.log(a)) + ((1-self.y) * np.log(1-a))) 
        return np.sum(logloss) / self.m
    
    def gradient(self, a):
        # dL/dw
        # dL/ db

        self.dw = np.zeros((self.n, 1)) # (n, 1)
        self.db = 0.0 # R

        # Find Error for Each Training Example
        for i in range(self.m):
            error_i = a[i] - self.y[i]
            self.db += error_i

            for j in range(self.n):
                self.dw[j] += error_i * self.X[i,j]
        
        self.dw /= self.m
        self.db /= self.m 

        return self.dw, self.db

    def gradient_v(self, a):

        error = a - self.y # (m, 1)

        # self.X.T.shape = (n, m) * (m, 1) --> (n, 1) 
        # Multiply each Row (Now the Same Feature for Different Observations) and Error, Sum All, and Put it to dw. (A Dot Product)
        self.dw = np.dot(self.X.T, error) 
        self.db = np.sum(error)

        self.dw /= self.m
        self.db /= self.m

        return self.dw, self.db

    def train(self, alpha=0.003, num_iters=1000):
        self.loss_h = []

        for i in range(num_iters):
            # Step 1: Compute the Linear Combination
            z = self.linear_combination_v()
            
            # Step 2: Apply the Sigmoid Function
            a = self.sigmoid(z)
            
            # Step 3: Compute the Gradients
            dw, db = self.gradient_v(a)
            
            # Step 4: Update Weights and Bias
            self.w -= alpha * dw
            self.b -= alpha * db
            
            # Compute and Store the Loss for This Iteration
            loss = self.J(a)
            self.loss_h.append(loss)
            
            # Optional: Print progress or check for convergence
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

        return self.w, self.b
    
    def plot_learning_curve(self):
        plt.plot(self.loss_h)
        plt.title('Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()



# Load the Data Set
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Initialize the Class
model = Perceptron(X, y)
w, b = model.train()
model.plot_learning_curve()