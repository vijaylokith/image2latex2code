"""
Evaluate
"""

# Imports
import nltk
import distance
import numpy as np

# Main Functions
# Score Functions
def Score_ExactMatch(references, hypotheses):
    '''
    Score - Exact Matching
    '''
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def Score_BLEU4(references, hypotheses):
    '''
    Score - BLEU-4
    '''
    references = [[ref] for ref in references] # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(
        references, hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25)
    )
    return BLEU_4


def Score_EditDistance(references, hypotheses):
    '''
    Score - Edit Distance
    '''
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot

# Main Vars
SCORE_FUNCS = {
    "exact_match": Score_ExactMatch,
    "bleu4": Score_BLEU4,
    "edit_distance": Score_EditDistance
}