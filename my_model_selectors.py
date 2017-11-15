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
        ch.setLevel(logging.CRITICAL)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handlers to logger
        # logger.addHandler(ch)
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
        best_score, selected_model = float("-inf"), self.base_model(self.n_constant)

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                # The term −2 log L decreases with increasing model complexity (more parameters),
                #  whereas the penalties p log N increase with increasing complexity
                # BIC = −2 * log L + p * log N
                # l: likelihood of the fitted model
                # p: number of free parameters, I've found different ways of calculating
                #    [n^2 + 2n * ft - 1] or
                #    [n * (n-1) + 2n * ft - 1] in case some points are fixed after one selection
                # n: number of data points

                # self.X.shape[1] contains the number of features

                model = self.base_model(n)
                log_l = model.score(self.X, self.lengths)
                # number of free parameters
                # p = n ** 2 + 2 * n * self.X.shape[1] - 1
                # bic_scorea = -2 * log_l + p * np.log(self.X.shape[0])

                p = n * (n-1) + (n-1) + 2 * self.X.shape[1] * n
                bic_score = (-2 * log_l) + (p * np.log(self.X.shape[0]))
                self.log.info("BIC: SCORE {} with n_features {} logN {} p {}"
                              .format(bic_score, self.X.shape[1], self.X.shape[0], p))
                if bic_score < best_score:
                    self.log.info("BIC: Old score {} was dethroned by score {} with {} components"
                                  .format(best_score, bic_score, n))
                    best_score, selected_model = bic_score, model
            except Exception as e:
                self.log.warn('BIC: EXCEPTION {}'.format(e))
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
        scores = []
        for w in words:
            score = model.score(w[0], w[1])
            self.log.debug("DIC: anti_l scores w0 len {}  w1 {} score {}".format(len(w[0]), w[1], score))
            scores.append(score)
        return scores # [model.score(w[0], w[1]) for w in words]

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score, selected_model = float("-inf"), self.base_model(self.n_constant)

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
                total_anti_log_l = self.log_l_list(model, all_words_but_i)
                # DIC = log_l - 1/(M-1)SUM(anti_log_l)
                average_anti_log_l = sum(total_anti_log_l)/len(all_words_but_i)
                dic_score = log_l - average_anti_log_l
                self.log.info("DIC: current score {} with n={} and log_l".format(dic_score, n, log_l))
                if dic_score > best_score:
                    self.log.info("DIC: Old score {} was dethroned by score {} with {} components"
                                  .format(best_score, dic_score, n))
                    best_score, selected_model = dic_score, model

            # if number of parameters exceed the number of samples
            except Exception as e:
                self.log.warn('DIC: EXCEPTION {}'.format(e))
                continue

        return selected_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score, selected_model = float("-inf"), self.base_model(self.n_constant)

        for n in range(self.min_n_components, self.max_n_components + 1):
            if len(self.sequences) < 2:
                hmm_model = self.base_model(n)
                log_likelihood = hmm_model.score(self.X, self.lengths)
                self.log.info('CV: Not enough sequences, we calculate pure logL {} with n={})'.format(log_likelihood, n))
            else:
                splits = KFold(min(3, len(self.sequences))).split(self.sequences)
                scores = []
                for train, test in splits:
                    train_X, train_lengths = combine_sequences(train, self.sequences)
                    test_X, test_lengths = combine_sequences(test, self.sequences)
                    try:
                        hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                        scores.append(hmm_model.score(test_X, test_lengths))
                    except Exception as e:
                        self.log.warn('CV: EXCEPTION {}'.format(e))
                        continue
                log_likelihood = np.average(scores) if len(scores) > 0 else float("-Inf")
                self.log.info('CV: logL {} with n={})'.format(log_likelihood, n))

            if log_likelihood > best_score:
                self.log.info("CV: Old score {} was dethroned by score {} with {} components"
                         .format(best_score, log_likelihood, n))
                best_score, selected_model = log_likelihood, self.base_model(n)

        return selected_model
