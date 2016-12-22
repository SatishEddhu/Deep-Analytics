# Dictionary is a mutable object
map1 = {"key1":10, "key2":20, "key3":30}
type(map1) # dict
print map1
map1.keys()

# Two ways of getting values from map
map1.get("key3")
map1["key3"]


# modifying map has a concise syntax
map1["key4"] = 70 # adding new key
map1["key2"] = 90 # can also modify values of existing keys

# No error thrown
print map1.get("key7") # 'None' is output instead of NULL; # "None' is a special keyword
type(map1.get("key7")) # NoneType
# Error is thrown
map1["key7"] # throws KeyError


# Iterate through map
for x in map1.keys():
    print x, map1.get(x)
        
for x in map1.iteritems():
    print x, type(x) # type(X) is a tuple
    
type(map1.iteritems()) # dictionary-itemiterator
