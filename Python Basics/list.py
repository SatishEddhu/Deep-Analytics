list1 = [30,20,10,40]
print list1
type(list1) # list

list2 = range(1,10,1)
type(list2)
help(range)

# Python indexing starts from 0 (R indexing starts from 1)
list1[0]
list1[0:2] # from 0 to n-1 = 1. Using 'n-1' to avoid confusion between indexing and count
list1[0:]
list1[:3]
list1[0::2] # :: to jump in strides. gets 0th and 2th characters
list1[0] = 100
list1

list3 = []
print list3
list3.append(10)
list3.append(20)
list3.insert(1,77)
list3.append(True) # list can have a heterogenous mix of value types
list3.append(list1)
print list3

list1.sort()
print list1
list1.count()

# There is no 'len' method in list
# Procedural style used here perhaps because len() is a common function for many types
len(list1) # procedural programming style
# invalid to add numeric to list. Only a list can be concatendated to a list
#list1 = list1 + 1

# no curly braces. Instad, use : and indentation
for x in list1:
    print x

list4 = [100,200]
x,y = list4 # this feature useful for tuples but not so useful in a list??
print x
print y
