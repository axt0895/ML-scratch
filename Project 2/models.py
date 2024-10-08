import numpy as np

class LDAModel:
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.shared_cov = None
        self.shared_cov_inv = None
        self.num_class = 0
        self.num_features = 0
    
    def fit(self, X, y):
        """
        Train the Linear Discriminant Analysis on the train data
        X: array-like of shape (num_samples, channels * height * width)
        y: array-like of shape (num_samples, )
        """
        
        #indentify the number of classes
        self.classes = np.unique(y)
        self.num_class = len(self.classes)
        self.num_features = X.shape[1]
        
        # calculate the class priors
        #calculate the mean for each class
        for c in self.classes:
            self.class_priors[c] = np.mean(y == c)
            self.class_means[c] = np.mean(X[y == c], axis = 0)
            
            
        #calculate the shared covariance
        self.shared_cov = np.zeros((self.num_features, self.num_features))
        for c in self.classes:
            class_cov = np.cov(X[y == c].T)
            self.shared_cov += self.class_priors[c] * class_cov
        self.shared_cov_inv = np.linalg.inv(self.shared_cov)
        
                                    
    def predict(self, X):
        '''
        Predicts the label for the provided test data
        X: array-like of shape (num_samples, num_features)
            The input data (RGB images) reshape to (num_samples, num_features)
        Returns
        -------
        predictions: array-like of shape(num_samples, )
            The predicted test labels
        '''
        
        discriminant_scores = []
        for c in self.classes:
            prior = self.class_priors[c]
            mean = self.class_means[c]
            
            #calculate the linear term
            linear_term = X @ self.shared_cov_inv @ mean.T
            
            #calculate the quadratic term
            quadratic_term = -0.5 * mean.T @ self.shared_cov_inv @ mean
            
            #log prior term
            log_prior = np.log(prior)
        
            #discriminant score
            score = linear_term + quadratic_term + log_prior
            discriminant_scores.append(score)
        
        discriminant_scores = np.array(discriminant_scores)
        
        #predic the class with highest discriminant score
        predictions = self.classes[np.argmax(discriminant_scores, axis = 0)]
        return predictions
        
            
class QDAModel:
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.class_cov = {}
        self.class_cov_inv = {}
        self.num_class = 0
        self.num_features = 0
    
    def fit(self, X, y):
        """
        Fit the QDA model on the provided data
        X: array-like of shape (num_samples, channels * height * width)
           The input data (RGB images) 
        y: array-like of shape (num_samples, )
            The class labels for the train data
        """
        self.classes = np.unique(y)
        self.num_class = len(self.classes)
        self.num_features = X.shape[1]
        
        #calculate the class priors
        for c in self.classes:
            self.class_priors[c] = np.mean(y == c)
            
        
        #calculate the class means
        #calculate the class specific covariance
        #precompute the inverse of covariance of each class 
        for c in self.classes:
            self.class_means[c] = np.mean(X[y == c], axis = 0)
            self.class_cov[c] = np.cov(X[y == c].T)
            self.class_cov_inv[c] = np.linalg.pinv(self.class_cov[c])

            
        
    def predict(self, X):
        """
        Predicts the class labels for the test data
        X: array-like of shape (num_samples, channels * height * width)
            The input data (RGB images)
        y: array-like of shape (num_sample, )
            The class labels for the test data
        """
        discriminant_scores = np.zeros((self.num_class, X.shape[0]))
        for c in self.classes:
            priors = self.class_priors[c]
            mean = self.class_means[c]
            cov_inv = self.class_cov_inv[c]
            sign, log_cov_det = np.linalg.slogdet(self.class_cov[c])
            
            #quadratic term
            quadratic_term = -0.5 * np.sum((X - mean) @ cov_inv * (X - mean), axis = 1)
            
            #log-prior term
            log_det_term = -0.5 * log_cov_det
            log_prior = np.log(priors)
            
            #discriminant scores
            score = quadratic_term + log_det_term + log_prior
            discriminant_scores[c] = score
            
        
        predictions = self.classes[np.argmax(discriminant_scores, axis = 0)]
        return predictions

        
class GaussianNBModel:
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}
        
        
    def fit(self, X, y):
        # TODO: Implement the fit method
        self.classes = np.unique(y)
        num_samples, num_features = X.shape
        
        for c in self.classes:
            self.class_priors[c] = np.mean(y == c)
            self.class_means[c] = np.mean(X[y == c], axis = 0)
            self.class_variances[c] = np.var(X[y == c], axis = 0) + 1e-8

    def predict(self, X):
        # TODO: Implement the predict method
        predictions = []
        for sample in X:
            class_scores = {}
            for c in self.classes:
                mean = self.class_means[c]
                variance = self.class_variances[c]
                prior = self.class_priors[c]
                
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) + ((sample - mean) ** 2) / variance)
                
                class_scores[c] = np.log(prior) + log_likelihood
                
                  # Get the class with the highest score
            predictions.append(max(class_scores, key = class_scores.get))

        return np.array(predictions)

                
    