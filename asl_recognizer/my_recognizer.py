import warnings
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
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    for test_word in test_set.get_all_Xlengths().keys():
        X_w, lengths_w =  test_set.get_all_Xlengths()[test_word]

        probs = {}
        best_score = None
        word_guess = None

        # log-likelihood is computed and word with the highest log-likelihood is selected
        for word_cand in models.keys():
            model = models[word_cand]
            try:
                score = model.score(X_w, lengths_w)
                probs[word_cand] = score

                if((best_score == None) or (score > best_score)):
                    best_score = score
                    word_guess = word_cand
            except:
                continue

        probabilities.append(probs)
        guesses.append(word_guess)

    return probabilities, guesses
