# list of built-in packages available currently
dir()

import math # to bring a package into the current session
dir() # math has been added to the list of packages in current session. R equivalent is ls()
dir(math) # to see details of 'math' package. R equivalent is ls(math)

math.log(100)

import math as m # to use 'm' as a short-hand for 'math'
m.log(100)
dir(m)

import numpy as np # numerical packages
dir(np) # big package and the basis for python

import sklearn as ml
dir(ml)
import matplotlib as mp
dir(mp)
