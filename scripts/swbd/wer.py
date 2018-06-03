"""WER."""
'''
@author: Hao Li
@email: lh575526@gmail.com
@data: 2018-06-03
'''
import math

def compute_wer(reference_corpus, translation_corpus, lower_case=False):
    """Compute wer of translation against references.

    Parameters
    ----------
    reference_corpus: list(list(str))
        references for each translation.
    translation_corpus: list(list(str))
        Translations to score.
    lower_case: bool, default False
        Whether or not to use lower case of tokens

    Returns
    -------
    the total WER
    TODO:
    5-Tuple with the Sub error, Del error, Ins error,
        WER, and SER
    """
    tot_distance = 0
    tot_ref_wc = 0
    for reference, translation in zip(reference_corpus, translation_corpus):
        if lower_case:
            reference = list(map(str.lower, reference))
            translation = list(map(str.lower, translation))

        tot_distance += _edit_distance(reference, translation)
        tot_ref_wc += len(reference)

    return tot_distance/tot_ref_wc

def _edit_distance(ref, hyp):
    """Compute edit distance of ref against hyp.

    Parameters
    ----------
    ref: list
        references 
    hyp: list
        hypothesis

    Returns
    -------
    the edit distance between two list
    """
    distance_matrix = [ [0 for j in range(len(hyp)+1)] for i in range(len(ref)+1) ]
    for i in range(len(ref)+1):
        distance_matrix[i][0] = i
    for j in range(len(hyp)+1):
        distance_matrix[0][j] = j

    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                distance_matrix[i][j] = distance_matrix[i-1][j-1]
            else:
                distance_matrix[i][j] = min(distance_matrix[i-1][j]+1, distance_matrix[i][j-1]+1, 
                                                distance_matrix[i-1][j-1]+1)

    return distance_matrix[len(ref)][len(hyp)]
