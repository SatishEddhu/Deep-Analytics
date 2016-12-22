tuple1 = (10,20,30,40) # immutable list. paranthesis instead of []
type(tuple1) # tuple
print tuple1

tuple1[0]
tuple1[0:2] # from 0 to n-1 = 1. Using 'n-1' to avoid confusion between indexing and count
tuple1[0:]
tuple1[:3]
tuple1[::2]
# append method not available in tuple as it is immutable

len(tuple1)

for x in tuple1:
    print x

tuple2 = (10,20)
x,y  = tuple2 # specially supported for tuples; to extract (x,y) values from a 'point'
print xs
print y
