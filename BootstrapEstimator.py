# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Single functions
from time import time
from sklearn.model_selection import train_test_split


class BootstrapEstimator:
    def __init__(self, classifier):

        ''' Class for estimating the accuracy of a given ML model
        by training a set of of models with bootstrapped datasets from a given
        dataset (with replacement).'''

        self.classifier = classifier

    def fit_calculate(self, X, y, acc_score, scaler = None,
            n = 1000, test_size = 0.25, frac = 1.0, random_state = None,
            stratify = None, verbose = False):

        ''' Method that fits models for bootstrapped data and calculates the
        'acc_score' for each bootstrapped data split into training and test
        sets. The 'acc_score' can be for a classifier
        such as sklearn.metrics.accuracy_score or a regressor
        such as sklearn.metrics.mean_squared_error.'''

        self.accuracy_score = acc_score
        self.results = {'train':[], 'test':[]}
        start = time()

        train_results = []
        test_results = []
        for i in range(1, n+1):

            # Generating bootstraps and defining a train/test split
            data = pd.DataFrame(X)
            data['target'] = y
            btsrp_data = data.sample(frac = frac, replace = True,
                                     random_state = i)
            X_train, X_test, y_train, y_test = train_test_split(
                btsrp_data.drop(['target'], axis = 1), btsrp_data['target'],
                test_size = test_size, stratify = stratify,
                random_state = random_state)

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
            train_results.append(self.accuracy_score(y_train, y_tr_pred))
            test_results.append(self.accuracy_score(y_test, y_te_pred))

            # Print the percentage done and the time remaining
            if verbose == True:
                mins = int((n-i)*(time()-start)/(i*60))
                secs = int((n-i)*(time()-start)/i)%60
                prc = int(100*i/(n+1))
                prm = int(10*(100*i/(n+1)-prc))
                print(('{}.{}% done, {} minutes and {} seconds remaining'+\
                       ' '*30).format(prc, prm, mins, secs), end = '\r')

        # Convert the results to numpy arrays and save to as attribute
        self.results['train'] = np.array(train_results)
        self.results['test'] = np.array(test_results)

        # End
        if verbose == True:
            print('done'+' '*100)

    def plot(self, bins = 20, conf = 0.9, stat = 'probability'):

        ''' A Method that plots histograms for the train and test results,
        respectively. The histograms also show the chosen confidence
        interval as a shaded region.
        '''

        self.bins = np.histogram(np.hstack((self.results['train'],
                                 self.results['test'])), bins = bins)[1] # edges

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
