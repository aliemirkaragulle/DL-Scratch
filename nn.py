from sklearn import datasets # type: ignore
import numpy as np

np.random.seed(42)

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

        # Initialize the Weights and the Bias
        self.w = np.zeros(self.n) # A Weight for each Feature in X
        self.b = 0.0
    
    def linear_combination(self):
        # z = w.T*X + b
        # A Linear Combination, wT.X + b for Each Observation (Rows)
        z = np.zeros(self.m)

        # For Each Feature Vector
        for i in range(self.m):
            # Get the Feature Vector and Calculate wT.X
            fv = self.X[i,:]
            z[i] = np.dot(self.w, fv)
        # w.T*X + b
        z += self.b
        z = z.reshape(-1,1) # Reshape z into a Column Vector to Match with the Dimensions of y

        return z 

    def linear_combination_v(self):
        z = np.dot(self.X, self.w) + self.b
        z = z.reshape(-1,1) # Reshape z into a Column Vector to Match with the Dimensions of y
        return z
    
    def sigmoid(self):
        z = self.linear_combination_v()
        return 1 / (1 + np.exp(-z))
       
    def J(self):
        # Binary Cross Entropy Loss/ Log Loss
        a = self.sigmoid()

        logloss = -((self.y * np.log(a)) + ((1-self.y) * np.log(1-a))) 
        return np.sum(logloss) / self.m
    
    def gradient(self):
        a = self.sigmoid()

        self.dw = np.zeros(self.n)
        self.db = 0.0

        # Find Error for Each Training Example
        for i in range(self.m):
            error_i = a[i] - self.y[i]
            self.db += error_i

            for j in range(self.n):
                self.dw[j] += error_i * self.X[i,j]
        
        self.dw /= self.m
        self.db /= self.m 

        return self.dw, self.db


    def gradient_v(self):
        a = self.sigmoid()

        error = a - self.y # (m, 1)

        # self.X.T.shape = (n, m) * (m, 1) --> (n, 1) 
        # Multiply each Row (Now the Same Feature for Different Observations) and Error, Sum All, and Put it to dw. (A Dot Product)
        self.dw = np.dot(self.X.T, error) 
        self.db = np.sum(error)

        self.dw /= self.m
        self.db /= self.m

        return self.dw, self.db

    def train(self, alpha=0.05, num_iters=1000):        
        
        for i in range(num_iters):
            dw, db = self.gradient_v()
            self.w = self.w - (alpha * dw)
            self.b = self.b - (alpha * db)

            if i % 100 == 0:
                cost = self.J()
                print(f"Iteration {i}, Cost: {cost}")
        
        return self.w, self.b 



# Load the Data Set
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Initialize the Class
model = Perceptron(X, y)
w, b = model.train()