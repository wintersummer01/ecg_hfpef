import os
import sys

from config import LOG_ROOT

def setOutputDir(tag, print_out):
    i = 0
    path = LOG_ROOT + '/' + tag + f"_{i}"
    while os.path.isdir(path):
        i += 1
        path = LOG_ROOT + '/' + tag + f"_{i}"
    
    os.mkdir(path)
    if not print_out:
        sys.stdout = open(path + "/log.out", 'w')
        sys.stderr = open(path + "/err.out", 'w')
    return path