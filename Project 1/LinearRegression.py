import numpy as np
import json

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # TODO: Initialize the weights and bias based on the shape of X and y.
        
        
        #initialization of the parameters
        self.bias = np.random.randn(1)
        self.weights = np.random.randn(X.shape[1])
        parameters = np.concatenate([self.bias, self.weights])
  
        
        
        #preparing our X with the added intercept form
        X_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

        
        # TODO: Implement the training loop.
        no_samples = X.shape[0]
        no_batches = no_samples // self.batch_size
        learning_rate = 0.01
        best_loss = float('inf')
        epochs_without_improvement = 0
        loss_history = []
   
        for epoch in range(self.max_epochs):
            
            indices = np.random.permutation(no_samples)
            X_shuffled = X_intercept[indices] 
            y_shuffled = y[indices]

            
            #loss for current epoch
            epoch_loss = 0
            
            for batch in range(no_batches):
                        
                start = batch * self.batch_size
                end = start + self.batch_size
                
                # batch of training and testing data
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                
                # Making the predictions using our randomly initialized parameters
                hypothesis = X_batch @ parameters
                
        
                
                # Computing the loss function
                loss = np.mean((hypothesis - y_batch) ** 2) + 0.5 * regularization * np.sum(parameters[1:] ** 2)
                loss_history.append(loss)
                
                epoch_loss += loss
                
                # Compute gradients
                gradients = X_batch.T @ (hypothesis - y_batch) / self.batch_size
                gradients[1:] += self.regularization * parameters[1:]

                # Update parameters
                parameters -= learning_rate * gradients
                
           
            epoch_loss /= no_batches

            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
        # models parameters
        
        self.bias = np.array(parameters[0])
        self.weights = np.array(parameters[1:])
        
        return loss_history
     
            

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        X_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

        parameters = np.concatenate([self.bias.reshape(-1), self.weights.reshape(-1)])
        
        # making the prediction
        hypothesis = X_intercept @ parameters
        
        return hypothesis
        
    
    

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # making  the prediction
        hypothesis = self.predict(X)
        
        mean_squared_error = np.mean((hypothesis - y) ** 2)
        
        return mean_squared_error