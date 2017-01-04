import numpy as np

a1 = np.array([[1,2,3], [4,5,6]])
a1.shape
a1[1,1]

# use colon to get values across dimensions
a1[1,:] # 2nd row, all columns
a1[:,1] # 2nd column, all rows

# colon also used for range
a1[0:2,1] # 1st and 2nd rows, 2nd column

# create zero matrix of size 2 x 3
a2 = np.zeros((2,3), int)
a2.shape
a2

# create ones matrix of size 3 by 2
a3 = np.ones((3,2), int)
a3.shape
a3

# create identity matrix of size 3 x 3
a4 = np.eye(3,3,dtype=int)
a4.shape
a4

# reshape the matrix
a5 = np.array([
       [1,2],
       [3,4],
       [5,6],       
])
a5
a5.reshape(2,3)
tmp = a5.reshape((1,6)) # 2D array only; not a 1D array
type(tmp)

# reshape a matrix into a single row
a6 = a5.reshape((1,-1))
# if we don't remember how many elements are present; "-1" means all elements
a6
type(a6)
a5.reshape(1,-1)

# get back the original original
a7 = a6.reshape(3,2)
a7

# getting useful statistics on matrix
a1
# 0 -> column statistics; 1 -> row statistics
a1.max(axis = 0) # max value from each column
a1.max(axis = 1) # max value from each row
a1.mean(axis = 1)
a1.std(axis = 0)

# element-wise operations on matrices
a7 = np.array([[1,2],[3,4]])
a8 = np.array([[1,1],[2,2]])
a7 + a8
a7 * a8 # element-wise multiplication

# matrix multiplication
a7.dot(a8) # dot product

# matrix transpose
a7.T
