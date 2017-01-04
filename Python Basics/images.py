from skimage import data # only import subset of a package
from skimage import io
from skimage import color
import os
import pandas as pd

# Tools> Preferenes> IPython Console> Graphics > Backend = Automatic (instead of inline) - restart
coffee = data.coffee()
type(coffee) # 400 x 600 x 3 -> Each of R,G,B have 400 x 600 values
coffee.shape
io.imshow(coffee)
coffee

# convert color image to black and white image
coffee_gray = color.rgb2gray(coffee)
type(coffee_gray)
coffee_gray.shape # 400 x 600 : 0 = white, 255 = black
io.imshow(coffee_gray)
coffee_gray

# reshape 400 x 600 2D to 240000 x 1 1D and again back to the original 2D shape
tmp1 = coffee_gray.reshape((1,-1))
type(tmp1)
tmp1.shape
tmp2 = tmp1.reshape((400,600))
tmp2.shape
io.imshow(tmp2)


os.getcwd()
os.chdir("D:\\Deep Analytics\\DigitRecognizer") # use \\ for windows/linux compatibility

digit_train = pd.read_csv("train.csv")
digit_train.shape

image = digit_train.iloc[32,1:] # extract a digit's image - 32nd row, 1st column onwards
image.shape
image_orig = image.reshape([28,28])
image_orig.shape
image_orig
image_orig = image_orig/255.0 # change to gray-scale of [0,1]
io.imshow(image_orig)
