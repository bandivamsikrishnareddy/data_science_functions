class AdaptiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_factor=1.5, strategy_mapping=None):
        self.outlier_factor = outlier_factor
        self.strategy_mapping = strategy_mapping or {
            "categorical": "most_frequent",
            "numerical_no_outliers": "mean",
            "numerical_outliers": "median"
        }
    def _detect_outliers(self, col):
        iqr = np.percentile(col, 75) - np.percentile(col, 25)
        upper = np.percentile(col, 75) + self.outlier_factor * iqr
        lower = np.percentile(col, 25) - self.outlier_factor * iqr
        return col[(col < lower) | (col > upper)]
    def _impute(self, col, strategy):
        if strategy == "passthrough":
            return col
        elif strategy == "most_frequent":
            return col.mode().iloc[0]  # Handle potential ties
        elif strategy == "mean":
            return col.mean()
        elif strategy == "median":
            return col.median()
        else:
            raise ValueError(f"Invalid imputation strategy: {strategy}")
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_transformed = X.copy()
        for col_name in X_transformed.columns:
            col = X_transformed[col_name]
            # Handle missing values
            if col.isnull().sum() == 0:
                strategy = "passthrough"
            else:
                # Determine data type
                if isinstance(col, pd.CategoricalDtype):
                    strategy = self.strategy_mapping["categorical"]
                else:  # Numerical data
                    outliers = self._detect_outliers(col)
                    if len(outliers) / len(col) > 0.25:
                        strategy = self.strategy_mapping["numerical_outliers"]
                    else:
                        strategy = self.strategy_mapping["numerical_no_outliers"]
            # Impute missing values
            X_transformed.loc[:, col_name] = self._impute(col, strategy)
        return X_transformed
