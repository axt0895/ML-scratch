import numpy as np

class LogisticsRegression:
    
    def __init__(self, num_class = 3, batch_size = 32, regularization = 0, max_epochs = 100, patience = 3):
        self.num_class = num_class
        self.batch_size = batch_size
        self.regularization = regularization
        self.batch_size = batch_size 
        self.max_epochs = max_epochs
        self.patience = patience
        self.bias = None
        self.weights = None
        
    def fit(self, X, y, num_classes = 3, batch_size = 32, regularization = 0, max_epochs = 100, patience = 3):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        
        
        # Initialization of weights & parameter
        self.bias = np.random.randn(1, self.num_class)
        self.weights = np.random.randn(X.shape[1], self.num_class)
        parameters = np.concatenate([self.bias, self.weights])

        # Preparing our X with the intercept form
        X_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        
        
        learning_rate = 0.01
        num_samples = X.shape[0]
        num_batches = X.shape[0] // self.batch_size
        best_loss = float('inf')
        epoch_without_improvement = 0
        epsilon = 1e-9
        loss_history = []
        y_encoded = np.eye(self.num_class)[y]
        
        
  
        # Implementation of the training loop
        for epoch in range(self.max_epochs):
            
            # Shuffling the data
            indices = np.random.permutation(num_samples)
            X_shuffled  = X_intercept[indices]
            y_shuffled = y_encoded[indices]
            epoch_loss = 0
            
            for batch in range(num_batches):
                start = batch * self.batch_size
                end = start + self.batch_size
                
                #Batch of training & testing data
                
                X_batch = X_shuffled[start: end]
                y_batch = y_shuffled[start: end]
                
                # Making the prediction on data
                Z = X_batch @ parameters
                exp_z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
                hypothesis = exp_z / np.sum(exp_z, axis = 1, keepdims = True)
                
                # Computing the loss function
                loss_per_sample = -np.sum(y_batch * np.log(hypothesis + epsilon), axis = 1)
                loss = np.mean(loss_per_sample)
                epoch_loss += loss
                loss_history.append(loss)
            
                #Compute the gradients
                gradients = X_batch.T @ (hypothesis - y_batch) / self.batch_size
                gradients += regularization * parameters
                
                # Updating the parameters
                parameters -=  learning_rate * gradients
                
                
            
            epoch_loss /= num_batches
            
            #Early Stopping
            if epoch_loss < best_loss:
                epochs_without_improvement = 0
                best_loss = epoch_loss
            else:
                if epochs_without_improvement >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
                epochs_without_improvement += 1
                
        self.bias = parameters[0,:]
        self.weights = parameters[1:,]
        return loss_history
    
    def predict(self, X):
        
        X_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        parameters = np.concatenate([self.bias.reshape(1, -1), self.weights])
        Z = X_intercept @ parameters
        exp_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
        hypothesis = exp_Z / np.sum(exp_Z, axis = 1, keepdims = True)
        
        y_prediction = np.argmax(hypothesis, axis = 1)
        
        return y_prediction
    
    
    def score(self, X, y):
        
        X_intercept = np.hstack([np.ones((X.shape[0], 1)),X])
        parameters = np.concatenate([self.bias.reshape(1, -1), self.weights])
        
        Z = X_intercept @ parameters
        exp_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
        hypothesis = exp_Z / np.sum(exp_Z, axis = 1, keepdims = True)
        
        y_encoded = np.eye(self.num_class)[y]
        
        score = -np.sum(y_encoded * np.log(hypothesis)) / X.shape[0]
        
        return score
        