# consistent indentation identifies the function body in Python
def abs(x):
    if x < 0: return -x
    return x

abs(5)
abs(-5)

# default parameter values
def add(x, y=1):
    return x + y
    
add(5)
add(5,5)

def test(x,y=1,z=0):
    return (x+z)*y

test(10)
test(10, 10) # 2nd argument value assigned to y
test(x=10,z=10) # using named arguments; 2nd argument assigned to z

# lambda is an anonymous function (no function name)
# (x,y) : return x + y
# no explicit return statement in lambda
# lambda useful only with short code; otherwise use 'def'
add1 = lambda x,y:x+y
add1(10,20)

# find squares of a bunch of numbers
def square(x):
    return x*x

# Approach 1
for i in range(1,10):
    print square(i)

# Approach 2
# equivalent to lapply in R
map(square, range(1,10))

# Approach 3 - shows real power of lambda since we don't even have
# to assign a name for the anonymous lambda function
map(lambda x:x*x, range(1,10))
# Below longer code not needed
# sqr = lambda x:x*x
# map(sqr, range(1,10))


# Comprehensions
# --------------
# To comprehend complex logic in a simple way

# generate a map of numbers and their squares
squares1 = {} # '{' means dictionary
for i in range(1,10):
    squares1[i] = i*i

# same output as above code but shorter and more readable
# 'comprehension' syntax
squares2 = {x: x**2 for x in range(1,10)}
squares2

import functools

# how to pass 'optional' arguments to functions when used with 'map' - use functools!
a = [10,20,30]
map(add, a) # fine till here

# map(add(y=2), a) # gives error message
# perhaps supporting this could have made Python implementation complicated

# This is the right approach
map(functools.partial(add,y=2), a)

map(functools.partial(test,z=2,y=3), a)
