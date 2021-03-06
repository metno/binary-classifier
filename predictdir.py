#!/usr/bin/env python3

#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''
# vi: syntax=python



import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import math
import predictor
import argparse




#cpdir = './cc-predictor-model'
#cpdir = './modeldata'

parser = argparse.ArgumentParser(description='Do cloud coverage preditcion on image')
parser.add_argument('--dirname', type=str, help='Input directory with imagesto do prediction on')
parser.add_argument('--modeldir', type=str, help='Model dir', default='modeldata')
parser.add_argument('--epoch', type=str, help='epoch', default=888)
parser.add_argument('--with-probs', type=bool, default=False, help='output probabilities')
args = parser.parse_args()


predictor = predictor.Predictor(args.modeldir, int(args.epoch))


def calc_spread(vector):
    i = np.array([0, 1]) / 2
    x = np.array(vector)
    mean = np.sum(x * i)
    variance = sum(i * i * x) - mean*mean
    return variance

if __name__ == "__main__":

    for filename in glob.iglob( args.dirname + '/**/*.jp*g', recursive=True):
    
        result = predictor.predict(filename)
        sys.stdout.write(filename + " ")

        if isinstance(result, (list, tuple, np.ndarray)):
            probs = np.argmax(result[0]) # Array of probabilities
            sys.stdout.write("%d" % probs)
            if args.with_probs:
                sys.stdout.write(" probabilities: [ ")
                for p in probs[0]:
                    sys.stdout.write("%0.2f%%, " % (p*100.0))
                sys.stdout.write(" ]")
                sys.stdout.write(" spread: %.02f" % calc_spread(result[0]))
            print("")
        else:
            print("ERROR: %s: %d " % (filename,result))  # Error


