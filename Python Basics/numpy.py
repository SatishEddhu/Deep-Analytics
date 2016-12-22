import numpy as np

height = [10,20,30]
weight = [100,200,300]

type(height)

# vectorized operations not supported in list
height + weight # concatenates vectors instead of doing pair-wise addition
height ** 2 # power of 2 (Squared) not supported
height > 20

# transform list to array
np_height = np.array(height)
np_height
np_weight = np.array(weight)
np_weight

np_height[0]
np_height[-1] # 1st element in reverse direction
np_height[0:2] # sliced array
np_height[0:2:2] # jump by 2 positions; index 2-1=1 is crossed after 1st jump itself

# vectorized operations supported on 'numpy' arrays
np_height + np_weight
np_height ** 2
np_height > 20
np_height[np_height > 15]

# list is heterogenous; array is homogenous
list1 = [10, True, 'abc']
type(list1)
list1
# casting heterogenous list to array; everything is cast to a string
array1 = np.array(list1)
array1
array1 + array1 # not supported since type is 'S11'. 'str' types would have been concatenated

np.mean(height)
np.mean(np_height)

a1 = np.array([[1,2,3], [4,5,6]])
a1[1,1]
a1[1,] # 2nd row displayed
a1[,1] # Error; not supported
