"""
Formula OCR
"""

# Imports
# Model Imports
from Library.Pix2Tex import FormulaOCR_Pix2Tex
from Library.GAN_TeacherForcing import FormulaOCR_GAN_TeacherForcing
# Dataset Imports
from Data.Datasets.GeneratedDatasets import DatasetUtils as DatasetUtils_GeneratedDatasets
from Data.Datasets.IM2LATEX_100K import DatasetUtils as DatasetUtils_IM2LATEX_100K
from Data.Datasets.FinalTest import DatasetUtils_Short as DatasetUtils_FinalTest_Short
from Data.Datasets.FinalTest import DatasetUtils_Long as DatasetUtils_FinalTest_Long

# Main Functions
def LaTeXParse_Tokenize(instring, prefix="", tokens=list(FormulaOCR_GAN_TeacherForcing.OCR["tokenizer"].get_vocabulary())):
    '''
    Tokenize a LaTeX string
    '''
    # Base Case
    if not instring: return []
    # Set Words
    words = set(tokens)
    # Tokenize
    if (not prefix) and (instring in words):
        return [instring]
    prefix, suffix = prefix + instring[0], instring[1:]
    solutions = []
    # Case 1: prefix in solution
    if prefix in words:
        try:
            solutions.append([prefix] + LaTeXParse_Tokenize(suffix, "", words))
        except ValueError:
            pass
    # Case 2: prefix not in solution
    try:
        solutions.append(LaTeXParse_Tokenize(suffix, prefix, words))
    except ValueError:
        pass
    if solutions:
        return sorted(solutions,
                      key = lambda solution: [len(word) for word in solution],
                      reverse = True)[0]
    else:
        raise ValueError("No Solution")

# Main Vars
OCR_MODULES = {
    "Pix2Tex": FormulaOCR_Pix2Tex.PIX2TEX_FUNCS,
    "FormulaGAN": FormulaOCR_GAN_TeacherForcing.PIX2TEX_FUNCS
}

DATASETS = {
    "GeneratedDatasets": DatasetUtils_GeneratedDatasets,
    "IM2LATEX_100K": DatasetUtils_IM2LATEX_100K,
    "FinalTest_Short": DatasetUtils_FinalTest_Short,
    "FinalTest_Long": DatasetUtils_FinalTest_Long
}

# RunCode
