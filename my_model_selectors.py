import math
import statistics
import warnings
import logging

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
        LOG_FILENAME = 'model_s.log'
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(LOG_FILENAME)
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)
        self.log = logger

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
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
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # fallback
        best_score, selected_model = float("inf"), self.base_model(self.n_constant)

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                # The term −2 log L decreases with increasing model complexity (more parameters),
                #  whereas the penalties p log N increase with increasing complexity
                # BIC = −2 * log L + p * log N
                # l: likelihood of the fitted model
                # p: number of parameters, I've found different ways of calculating
                #    [n^2 + 2n * ft - 1] or
                #    [n * (n-1) + 2n * ft - 1] in case some points are fixed after one selection
                # n: number of data points

                # self.X.shape[1] contains the number of features

                model = self.base_model(n)
                log_l = model.score(self.X, self.lengths)
                self.log.info("BIC: n_features {} vs {} "
                          .format(self.X.shape[1], sum(self.lengths)))
                # number of free parameters
                p = n ** 2 + 2 * n * sum(self.lengths) - 1
                self.log.info("BIC: logN {} vs {} vs {}"
                          .format(np.log(sum(self.lengths)), np.log(n), np.log(self.X.shape[0])))
                bic_score = -2 * log_l + p * np.log(self.X.shape[0])

                if bic_score < best_score:
                    self.log.info("BIC: Old score {} was dethronized by score {} with {} components"
                             .format(best_score, bic_score, n))
                    best_score, selected_model = bic_score, model
            except Exception as e:
                self.log.exception('BIC: EXCEPTION -> ', e)
                continue

        return selected_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    # calculate log likelihoods for a list of words
    def log_l_list(self, model, words):
        return [model.score(w[0], w[1]) for w in words]

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score, selected_model = float("inf"), self.base_model(self.n_constant)

        all_words_but_i = []
        for word in self.words:
            if word != self.this_word:
                all_words_but_i.append(self.hwords[word])

        for n in range(self.min_n_components, self.max_n_components+1):
            self.log.info("DIC: start with {}".format(n))
            try:
                model = self.base_model(n)
                # log(P(X(i))
                log_l = model.score(self.X, self.lengths)
                # anti log likelihoods
                # log(P(X(all but i)
                anti_log_l = self.log_l_list(model, all_words_but_i)
                # DIC = log_l - 1/(M-1)SUM(anti_log_l)
                dic_score = log_l - np.mean(anti_log_l)
                self.log.info("DIC: current score {} with n={}".format(dic_score, n))
                if dic_score > best_score:
                    self.log.info("DIC: Old score {} was dethronized by score {} with {} components"
                             .format(best_score, dic_score, n))
                    best_score, selected_model = dic_score, model

            # if number of parameters exceed the number of samples
            except Exception as e:
                self.log.exception('DIC: EXCEPTION -> ', e)
                continue

        return selected_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score, selected_model = float("inf"), self.base_model(self.n_constant)
        # define the number of folds we would like to use
        k_splits = 3
        for n in range(self.min_n_components, self.max_n_components + 1):
            self.log.info("CV: start with {}".format(n))
            if len(self.sequences) > 2:
                # in case we do not have enough data, let's say 2 is the minimum
                split_method = KFold(n_splits=min(k_splits, len(self.sequences)))
                cv_cumulative_score = 0.0
                i = 0
                # Cross validation loop, here we add logl
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    X_train, length_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test, length_test = combine_sequences(cv_test_idx, self.sequences)
                    try:
                        # train a gaussian model
                        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                             random_state=self.random_state, verbose=False).fit(X_train, length_train)

                        # cross validate this model with the other portion of the sequences and keep track of its score
                        cv_cumulative_score += float(model.score(X_test, length_test))
                        i += 1
                        self.log.info("CV: at indexes train {} and test {} we heve cumulative score of {}"
                                 .format(cv_train_idx, cv_test_idx, cv_cumulative_score))
                    except Exception as e:
                        self.log.exception('CV: EXCEPTION -> ', e)
                        continue
                # average the score over the number of contributions
                cv_score = cv_cumulative_score / i if i > 0 else float("-Inf")
                if cv_score > best_score:
                    self.log.info("CV: Old score {} was dethronized by score {} with {} components"
                             .format(best_score, cv_score, n))
                    best_score, selected_model = cv_score, self.base_model(n)

        return selected_model
