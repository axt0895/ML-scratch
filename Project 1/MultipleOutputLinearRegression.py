import numpy as np
import json

class MultipleLinearRegression:
    def __init__(self, batch_size = 32, regularization = 0, max_epochs = 100, patience = 3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        
    def fit(self, X, y, batch_size = 32, regularization =0, max_epochs= 100, patience = 3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        
        
        #intialization of parameters
        self.bias = np.random.randn(1, 2)
        self.weights = np.random.randn(X.shape[1], 2)
        parameters = np.vstack([self.bias, self.weights])
        
        # preparing our dataset with the added intercept form
        X_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        
        #implementing the training loop
        num_samples = X.shape[0]
        num_batches = num_samples // self.batch_size
        learning_rate = 0.01
        best_loss = float('inf')
        epochs_without_improvement = 0
        loss_history = []
        
        for epoch in range(self.max_epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_intercept[indices]
            y_shuffled = y[indices]
            
            #loss for current epoch
            epoch_loss = 0
            
            for batch in range(num_batches):
                start = batch * self.batch_size
                end = start + self.batch_size
                
                #batch of training & testing data
                X_batch = X_shuffled[start: end]
                y_batch = y_shuffled[start: end]
                
                # Making the predictions using our randomly intialized parameters
                hypothesis = X_batch[:, 1:] @ self.weights + self.bias
                
                #computing the loss function
                loss = np.mean(np.sum((hypothesis - y_batch) ** 2, axis = 1)) + 0.5 * regularization * np.sum(self.weights ** 2)
                loss_history.append(loss)
                
                
                #epoch_loss
                epoch_loss += loss
                
                #compute gradients
                dW = (2 / self.batch_size) * X_batch[:, 1: ].T @ (hypothesis - y_batch) + regularization * self.weights
                db = (2 / self.batch_size) * np.sum(hypothesis - y_batch, axis=0, keepdims=True)

                #Update the parameters
                self.weights -= learning_rate * dW
                self.bias -= learning_rate * db
                
            epoch_loss /= num_batches
            
            #Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else: 
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
                    
        # model parameters
        self.bias = np.array(parameters[0])
        self.weights = np.array(parameters[1: ])
        
        return loss_history
    
    def predict(self, X):
        
        X_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_intercept[:, 1:] @ self.weights + self.bias
    
    
    def score(self, X, y):
        
        #making the prediction
        hypothesis = self.predict(X)
        
        # Compute mean squared error
        mean_squared_error = np.mean(np.sum((hypothesis - y) ** 2, axis=1))
        
        return mean_squared_error