import os
import sys

from config import LOG_DIR

def setOutputDir(tag, print_out):
    i = 0
    path = LOG_DIR + '/' + tag + f"_{i}"
    while os.path.isdir(path):
        i += 1
        path = LOG_DIR + '/' + tag + f"_{i}"
    
    os.mkdir(path)
    if not print_out:
        sys.stdout = open(path + "/log.out", 'w')
        sys.stderr = open(path + "/err.out", 'w')
    return path

def toHfpefScore(criteria):
    BMI = criteria[:,0]
    afib = criteria[:,1]
    PASP = criteria[:,2]
    age = criteria[:,3]
    EE = criteria[:,4]
    
    score = 2*BMI + 3*afib + PASP + age + EE
    return score