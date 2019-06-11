import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    The BIC score is given by BIC = - 2* log(L) + p * log(N) where
    p: number of parameters = n*n + 2*n*m - 1 (n:# of hidden units, m:# of features)
    N: number of data points
    Under this definition of BIC, smaller the BIC score is, better the model is.
    (Note that the definition of BIC here is (-2) time the definition of BIC in
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    )
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on BIC scores
        best_num_components = None
        best_score = None
        best_model = None

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                n_data = self.X.shape[0] # number of data points
                n_features = self.X.shape[1] # number of features
                p = n * n + 2 * n * n_features - 1 # number of parameters
                score =  -2.0*logL + float(p) * float(np.log(n_data)) # BIC score

                if((best_score == None) or (score < best_score)):
                    best_num_components = n
                    best_score = score
                    best_model = model
            except:
                continue

        if(best_score != None):
            return best_model
        else:
            # print("failed to find the best model. base model with n={} returned.".format(self.n_constant))
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)\sum_{j != i}(log(P(X(j))
    Under this definition of DIC, the model with a larger DIC score is better.
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on DIC scores
        best_num_components = None
        best_score = None
        best_model = None

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)

                scores = []
                for word in self.hwords.keys():
                    X_w, lengths_w = self.hwords[word]

                    if word != self.this_word: # for computing the 2nd term (penalty term) of DIC score
                        logL = model.score(X_w, lengths_w)
                        scores.append(logL)
                    else: # for computing the 1st term of DIC score
                        logL_this_word = model.score(X_w, lengths_w)

                score = logL_this_word - np.mean(scores) # DIC score

                if((best_score == None) or (score > best_score)):
                    best_num_components = n
                    best_score = score
                    best_model = model
            except:
                continue

        if(best_score != None):
            return best_model
        else:
            # print("failed to find the best model. base model with n={} returned.".format(self.n_constant))
            return self.base_model(self.n_constant)



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds.
    A model with a larger average score is better.
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection using CV
        split_method = KFold(n_splits = 2)
        best_num_components = None
        best_score= None
        best_model = None

        for n in range(self.min_n_components, self.max_n_components+1):
            count = 0
            score = 0.0
            scores = []
            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    count += 1
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                    model = self.base_model(n)
                    logL = model.score(X_test, lengths_test)
                    scores.append(logL)

                    score = np.mean(scores) # average score for cross-validation folds

                    if((best_score == None) or (score > best_score)):
                        best_num_components = n
                        best_score = score
                        best_model = model

            except:
                continue

        if(best_score != None):
            return best_model
        else:
            # print("failed to find the best model. base model with n={} returned.".format(self.n_constant))
            return self.base_model(self.n_constant)
