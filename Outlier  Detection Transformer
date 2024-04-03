class OutlierRemover(BaseEstimator,TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def outliers_iqr(self, X, y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (iqr * self.factor)
        upper_bound = q3 + (iqr * self.factor)
        # filter only those rows that are greater than lower_bound and less than upper_bound, 
        #i.e. drop values outside the given interval
        X.loc[((X >= lower_bound) | (X <= upper_bound))] 
        
        return pd.Series(X)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.apply(self.outliers_iqr)
