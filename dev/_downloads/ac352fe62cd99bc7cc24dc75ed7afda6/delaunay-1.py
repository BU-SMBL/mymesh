import sys, os, time, subprocess
import scipy
import numpy as np
import matplotlib.pyplot as plt

def run_no_numba(npoints, seed):
    code = f"""