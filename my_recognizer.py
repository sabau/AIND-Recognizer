import warnings

import logging

from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    LOG_FILENAME = 'recognizer.log'
    log = logging.getLogger('Recognizer')
    fh = logging.FileHandler(LOG_FILENAME)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # Iteate through the test set where i represent the word we are analyzing
    for i in range(0, len(test_set.get_all_Xlengths())):
        test_X, test_lengths = test_set.get_item_Xlengths(i)
        log_l_dict = {}
        best_score, best_word = float('-Inf'), None

        # try to calculate the probabilities for each word/model and populate the dictionary
        for word, model in models.items():
            try:
                # Try to get the log likelihood of test_X for the current model
                score = model.score(test_X, test_lengths)
            except Exception as e:
                log.warn('EXCEPTION {}'.format(e))
                # We add this word to maintain the structure of the dictionary,
                # with probability 0
                score = float('-Inf')
            log_l_dict[word] = score
            log.info("Step {}: logl for word {} is {}".format(i, word, score))
            # Keep track of the most likely word
            if score > best_score:
                log.info("Old score {} for word {} was dethroned by score {} with {} word"
                         .format(best_score, best_word, score, word))
                best_score, best_word = score, word
        # Add the whole dictionary to the probability list
        probabilities.append(log_l_dict)
        # store in the guesses the most likely word
        guesses.append(best_word)

    return probabilities, guesses
