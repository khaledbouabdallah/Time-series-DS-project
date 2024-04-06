import pandas as pd
import random 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

class TsMissing(object):
    """
    A class for handling missing values in time series data.
    
    Parameters:
    - data (pandas.DataFrame): The input time series data.
    - columns (list): The columns to consider for missing values. If None, all columns will be considered.
    - bins (int): The number of bins to use for creating intervals of missing values.
    """
    
    # List of supported imputation strategies
    _strategies = ['forward fill', 'backward fill', 'linear', 'MICE', 'knn' ] # 'mean',  'theta', 
    
    def __init__(self, data, columns=None, bins=10):
        """
        Initialize the TsMissing class.
        
        Args:
        - data (pandas.DataFrame): The input time series data.
        - columns (list): The columns to consider for missing values. If None, all columns will be considered.
        - bins (int): The number of bins to use for creating intervals of missing values (default: 10).
        """
        if columns is None:
            self.df = data.copy()
        else:
            self.df = data[columns].copy()
            
        self.n_bins = bins
        self.index_missing_values = self.get_missing_index()
        self.index_not_missing = self.get_not_missing_index()
        self.intervals_missing, self.intervals_lengths_missing = self.get_lengths_intervals(self.index_missing_values)
        self.intervals_not_missing, self.intervals_lengths_not_missing = self.get_lengths_intervals(self.index_not_missing)
        self.bins, self.bins_probabilities = self.intervals_lengths_histogram()
            
    def get_missing_index(self):
        """
        Create a dictionary to store for each column the index (timestamp) of the missing values.
        
        Returns:
        - dict: A dictionary where the keys are column names and the values are a list of index of missing values.
        """
        index_missing_values = {}
        for column in self.df.columns:
            index_missing_values[column] = self.df[self.df[column].isnull()].index
        return index_missing_values
    
    def get_not_missing_index(self):
        """
        Create a dictionary to store for each column the index (timestamp) of the not missing values.
        
        Returns:
        - dict: A dictionary where the keys are column names and the are a values list of index of not missing values.
        """
        index_not_missing  = {}
        for column in self.df.columns:
            index_not_missing[column] = self.df[column].dropna().index
        return index_not_missing
    
    def get_lengths_intervals(self, indexs):
        """
        Create a dictionary to store for each column the intervals of missing values and their length.
        
        Args:
        - indexs (dict): A dictionary where the keys are column names and the are a values list of indexs
        
        Returns:
        - dict: A dictionary where the keys are column names and the values are a binary list [start_index, end_index] of intervals
        - dict: A dictionary where the keys are column names and the values are the lengths of the intervals.
        """
        
        intervals_lengths = {}
        intervals_dict = {}
        
        # iterate over the columns
        for key, value in indexs.items():
            intervals = []
            lengths = []
            start = value[0]

            # building the intervals
            for i in range(1, len(value)):
                if value[i] - value[i-1] != pd.Timedelta('1 days'):
                    intervals.append((start, value[i-1]))
                    start = value[i]
            # calculate the lengths of the intervals
            for interval in intervals:
                lengths.append((interval[1] - interval[0] + pd.Timedelta('1 days')).days)
                
            intervals_lengths[key] = lengths
            intervals_dict[key] = intervals
            
        return  intervals_dict, intervals_lengths
        
    def intervals_lengths_histogram(self):
        """
        Create a dictionary to store for each column the bins of the intervals of missing values and their probabilities.
        
        Returns:
        - dict: A dictionary where the keys are column names and the values are the bins of the intervals of missing values.
        - dict: A dictionary where the keys are column names and the values are the probabilities of the interval lengths.
        """
        lengths_probabilities = {}
        bins_intervals_sizes = {}
        
        for column in self.df.columns:
            hist, bins = np.histogram(self.intervals_lengths_missing[column], bins=self.n_bins)
            bins = [int(x) for x in bins]
            probabilities = hist / np.sum(hist)
            lengths_probabilities[column] = probabilities
            bins_intervals_sizes[column] = list(bins)
            
        return bins_intervals_sizes, lengths_probabilities
        
    def generate_test_intervals(self, column, test_size=30, random_state=None, max_iteration=1000, strategy='real', return_sizes=False):
        """
        Generate test intervals with missing values.
        
        Args:
        - df (pandas.DataFrame): The input time series data.
        - column (str): The column name to generate test intervals for.
        - test_size (int): The number of test intervals to generate.
        - random_state (int): The random seed for reproducibility.
        - max_iteration (int): The maximum number of iterations to find non-overlapping intervals.
        - strategy (str): The strategy to use for generating the test intervals. The options are 'real' and 'uniform'.
        - sizes (bool): If True, return the sizes of the test intervals.
        
        Returns:
        - list: A list of test intervals with missing values.
        """
        
        # set the random seed
        if random_state is not None:
            random.seed(random_state)
        
        test_gaps = []
        test_sizes = []
        i = 0

        while i < test_size:       
            # select the interval size
            if strategy == 'real':
                interval_size = random.choices(population=self.bins[column][:-1], weights=self.bins_probabilities[column])[0]
            elif strategy == 'uniform':
                interval_size = random.choices(population=self.intervals_lengths_missing[column])[0]
                
            test_sizes.append(interval_size)
            
            j = 0
            while j < max_iteration+1:
                j += 1
                
                # select the start index of the test missing interval
                interval = random.choice(self.intervals_not_missing[column])
                start_index = random.choice(self.df.loc[interval[0]:interval[1]].iloc[:10].index)
                
                # create the missing test interval
                missing_interval = (start_index, start_index + pd.Timedelta(days=interval_size))
                #print(interval,missing_interval,missing_interval[1] > interval[1] , self._interval_is_overlapping(missing_interval, test_gaps))
                # check if the gap is larger than the data and if it is overlapping with the missing intervals
                if missing_interval[1] > interval[1] or self._interval_is_overlapping_(missing_interval, test_gaps):
                    continue 
                else:
                    break     
                          
            # if we can't find non-overlapping intervals after 1000 tries, raise an error
            if j > max_iteration:
                #raise TimeoutError(f"Couldn't find non-overlapping interval after {max_iteration} tries, try to increase the max_iteration parameter")
                continue
                
            test_gaps.append(missing_interval)
            i += 1  
         
        if return_sizes:
            return self._flatten_intervals_(test_gaps), test_sizes
        else:    
            return self._flatten_intervals_(test_gaps)
        
    def _interval_is_overlapping_(self, interval, intervals):
        """
        Check if an interval is overlapping with a list of intervals.
        
        Args:
        - interval (list): An interval [start, end].
        - intervals (list): A list of intervals [start, end].
        
        Returns:
        - bool: True if the interval is overlapping with any of the intervals in the list, False otherwise.
        """
        
        for start, end in intervals:
            if interval[0] <= end and interval[1] >= start: 
                return True
        return False
    
    
    def _flatten_intervals_(self, intervals):
        """
        Flatten a list of intervals.
        
        Args:
        - intervals (list): A list of lists of intervals [start, end].
        
        Returns:
        - list: A list of timestamps.
        """
        
        data = []
        for start, end in intervals:
            data.extend(pd.date_range(start=start, end=end, freq='D'))
              
        return data

    
    def impute_all(self, test, metric = 'mae', random_state=42):
        
        # Create a dataframe to store the scores of the different imputation methods
        score_df = pd.DataFrame(columns=["strategy"] + self.df.columns.tolist())
        # Create a dataframe to store the missing values
        df_missing = self.df.copy()
        
        # Create a dictionary to store the original data minus the missing test data
        self.original_minus_null_index = {}
        # Create a dictionary to store the true values of the missing values
        y_true = {}   
        y_pred = {}
        # Create a dictionary to store the data after imputation
        df_pred_dict = {}
        
          
        
        for column in self.df.columns:
            # Store the index of the original data minus the missing test data
            self.original_minus_null_index[column] = list(set(self.df[column].loc[self.index_not_missing[column]].index).difference(set(test[column])))
            # Store the true values of the missing values
            y_true[column] = self.df[column].loc[test[column]]
            # Set the values of the missing test values to NaN
            df_missing[column].loc[test[column]] = np.nan


        for strategy in self._strategies:
            
            row = {"strategy": strategy}             
           
            
            if strategy == 'forward fill':
                df_pred = df_missing.ffill()
                
            elif strategy == 'backward fill':
                df_pred = df_missing.bfill()
                
            elif strategy == 'linear':
                df_pred = df_missing.interpolate(method='linear')
            
            elif strategy == 'knn':
                imputer = KNNImputer(n_neighbors=50)
                df_pred = pd.DataFrame(imputer.fit_transform(df_missing), columns=self.df.columns)
                df_pred.index = self.df.index
                
            elif strategy == 'MICE':
                imputer = IterativeImputer(random_state=random_state, n_nearest_features=5, max_iter=50, tol=0.001)
                df_pred = pd.DataFrame(imputer.fit_transform(df_missing), columns=self.df.columns)
                df_pred.index = self.df.index
                
            for column in self.df.columns:
                # Store the predicted values of the missing values
                y_pred = df_pred[column].loc[test[column]]  
                
                # Check if the imputation was successful in total
                if df_pred[column].isnull().sum() > 0:
                    print("Could not impute all data in column", column, "using the strategy", strategy)
              
                
                # Check if the imputation was successful for test
                if y_pred.isnull().sum() > 0:
                   print("Could not impute all test data in column", column, "using the strategy", strategy)
                   continue          
                                               
                # Calculate the score of the imputation
                score = self.evaluate(y_true[column], y_pred, metric=metric)
                # Store the score
                row[column] = score
            
            
            score_df = pd.concat([score_df, pd.DataFrame([row])], ignore_index=True)
            df_pred_dict[strategy] = df_pred.copy()
            
        return score_df, df_pred_dict, y_true, 
        
    def evaluate(self, y_true, y_pred,  metric= 'mae'):
        
        if metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'mse':
            return mean_squared_error(y_true, y_pred)
            
    
    def plot_missing(self): # @TODO
        pass
    
    def info(self): # @TODO
        pass
        
    def plot_imputed(self): # @TODO
        pass