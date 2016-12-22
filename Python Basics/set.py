# Set = unordered collection of distinct items
# has no indexed order; duplicates not allowed
# the {} syntax similar to dictionary/map, but has no key/value pair concept
set1 = {10,20,30,40}
type(set1)
set1.add(50)
set1[0] # Error - 'set' does not support indexing
set1

# we don't recommend using the __ type of methods
for x in set1.__iter__():
    print x

set2 = {10,10,30,40} # duplicate 10 is removed automatically
print set2
