# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
# Single functions
from time import time
from sklearn.model_selection import train_test_split

def train_test_split_ts(X, y, test_size = 0.2, n_splits = 3):

    ''' Train/test split for time ordered data, for which
        random sampling leads to data leakage. Function
        splits inputs "X" and target "y" into "n_splits"
        subsets. For each subset it selects the first
        (1-"test_size") samples as the train set and
        the final "test_size" samples as the test set.
        Inputs need to be be pandas.DataFrame or
        pandas.Series with pandas.Timestamp indexes.'''


    def valid_input(Z):

        ''' Check the input is an appropriate pandas
            timeseries. '''

        if (not isinstance(Z, pd.DataFrame)) and (not isinstance(Z,
                                                                 pd.Series)):
            raise TypeError('Inputs need to be pandas.DataFrame ' +
                             'or pandas.Series and indexes need to ' +
                             'be pandas.Timestamp')
        if (not (type(Z.index[0])) is pd.Timestamp):
            raise TypeError('Inputs need to be pandas.DataFrame ' +
                             'or pandas.Series and indexes need to ' +
                             'be pandas.Timestamp')

        # Some issues with 'isinstance' for pandas so not using it
        # for the second condition.

        return None

    _ = valid_input(X)
    _ = valid_input(y)

    X_splits = np.array_split(X.sort_index(), n_splits)
    y_splits = np.array_split(y.sort_index(), n_splits)

    test_idx = int(len(y_splits[0])*(1-test_size))
    X_train  = [split[:test_idx] for split in X_splits]
    y_train  = [split[:test_idx] for split in y_splits]
    X_test   = [split[test_idx:] for split in X_splits]
    y_test   = [split[test_idx:] for split in y_splits]

    return X_train, X_test, y_train, y_test

class BootstrapEstimator:
    def __init__(self, classifier):

        ''' Class for estimating the accuracy of a given ML model
            by training a set of of models with bootstrapped datasets
        f   rom a given dataset (with replacement).'''

        self.classifier = classifier

    def fit_calculate(self, X, y, acc_score, scaler = None, n = 1000,
        test_size = 0.25, frac = 1.0, random_state = None, stratify = None,
        verbose = False, time_series = False, n_splits = 4):

        ''' Method that fits models for bootstrapped data and
            calculates the 'acc_score' for each bootstrapped
            data split into training and test sets. The 'acc_score'
            can be for a classifier such as sklearn.metrics.accuracy_score
            or a regressor such as sklearn.metrics.mean_squared_error.'''

        self.results = {'train':[], 'test':[]}
        self.accuracy_score = acc_score

        def fitter(X_train, X_test, y_train, y_test, scaler = scaler):

            '''Fit the (bootsrapped) data to the classifier
               and calclate the accuracy score.'''

            # Using a scaler such as MinMax
            if scaler == None:
                X_tr_scaled, X_te_scaled = X_train, X_test
            else:
                scaler.fit(X_train)
                X_tr_scaled = scaler.transform(X_train)
                X_te_scaled = scaler.transform(X_test)

            # Fitting the classifier and calculating the scores
            self.classifier.fit(X_tr_scaled, y_train)
            y_tr_pred = self.classifier.predict(X_tr_scaled)
            y_te_pred = self.classifier.predict(X_te_scaled)

            return (self.accuracy_score(y_train, y_tr_pred),
                    self.accuracy_score(y_test, y_te_pred))

        # Loop n times
        start = time()
        train_results = []
        test_results = []
        random.seed(random_state)

        for i in range(1, n+1):

            # Generating bootstraps and defining a train/test split
            randstate = random.choice(range(n))
            data = pd.DataFrame(X)
            data['target'] = y
            btsrp_data = data.sample(frac = frac, replace = True,
                                     random_state = randstate)

            if time_series == False:
                X_train, X_test, y_train, y_test = train_test_split(
                    btsrp_data.drop(['target'], axis = 1),
                    btsrp_data['target'], test_size = test_size,
                    stratify = stratify, random_state = randstate)

                # Fit the classifier to a bootsrapped sample
                acc_train, acc_test = fitter(X_train, X_test, y_train, y_test,
                                             scaler = scaler)
                train_results.append(acc_train)
                test_results.append(acc_test)

            if time_series == True:
                X_train, X_test, y_train, y_test = train_test_split_ts(
                    btsrp_data.drop(['target'], axis = 1),
                    btsrp_data['target'], test_size = test_size,
                    n_splits = n_splits)

                # Loop over all n_splits subsets and fit classifier
                for j in range(n_splits):
                    acc_train, acc_test = fitter(X_train[j], X_test[j],
                                                 y_train[j], y_test[j],
                                                 scaler = scaler)
                    train_results.append(acc_train)
                    test_results.append(acc_test)

            # Print the percentage done and the time remaining
            if verbose == True:
                mins = int((n-i)*(time()-start)/(i*60))
                secs = int((n-i)*(time()-start)/i)%60
                prc = int(100*i/(n+1))
                prm = int(10*(100*i/(n+1)-prc))
                print(('{}.{}% done, {} minutes and {} seconds remaining'+\
                       ' '*30).format(prc, prm, mins, secs), end = '\r')

        # Convert the results to numpy arrays and save as attribute
        self.results['train'] = np.array(train_results)
        self.results['test'] = np.array(test_results)

        # End
        if verbose == True:
            print('done'+' '*100)

    def plot(self, bins = 20, conf = 0.9, stat = 'probability'):

        ''' A Method that plots histograms for the train and test results,
            respectively. The histograms also show the chosen confidence
            interval as a shaded region.'''

        # edges
        self.bins = np.histogram(np.hstack((self.results['train'],
                                 self.results['test'])), bins = bins)[1]

        def hist(data_set, color, shade, cl, ax, stat = stat):
            # data_set = 'train' or 'test'

            ''' Helper function for histogram plotting '''

            # Histogram from seaborn
            dat = self.results[data_set]
            sns.histplot(x = dat, stat = stat, ax = ax, bins = self.bins,
                         color = color)
            ylims = ax.get_ylim() # Store for later

            # Plot the mean
            M = np.mean(dat)
            label = r'mean = {:.4}'.format(M)
            ax.plot([M, M], [0,2*ylims[1]], color = 'k', linestyle = 'dashed',
                        label = label)

            # Confidence interval
            xvals = [cl[0], cl[0], cl[1], cl[1]]
            yvals = [0, 2*ylims[1], 2*ylims[1], 0]
            label = r'%{:.3} Conf. Int. = ({:.4}, {:.4})'.format(
                 conf*100, cl[0], cl[1])
            ax.plot(xvals, yvals, linestyle = 'dotted', label = label,
                dash_capstyle = "round", color = shade, alpha = 0.7)
            ax.fill_between(xvals, yvals, color = shade, alpha = 0.1)

            # Legend
            ax.legend(loc = 'upper center', ncol = 1,
                      bbox_to_anchor = (0.5, 1.25),
                      fancybox = True, shadow = True)

            # y axis range
            ax.set_ylim(ylims) # Cuts the yaxis to include only histogram

            # Axis labels
            ax.set_xlabel(
                self.accuracy_score.__name__+'({})'.format(data_set))
            ax.set_ylabel(stat)

        # Figure parameters
        plt.rcParams['font.size'] = '14'
        fig, ax = plt.subplots(nrows = 1, ncols = 2, sharey = True,
                               sharex = True, figsize = (15,5))
        fig.suptitle(self.classifier.__class__.__name__, fontsize = 20,
                         y = 1.14)

        # Two historgrams
        p = (1-conf)*50
        percs = list(np.percentile(self.results['train'], [p, 100-p]))
        hist('train', 'dodgerblue', 'blue', percs, ax[0])
        percs = list(np.percentile(self.results['test'], [p, 100-p]))
        hist('test', 'indianred', 'red', percs, ax[1])
        plt.show()
