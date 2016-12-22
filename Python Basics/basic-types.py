x = 15
print x
print type(x)
x = x+10 # 25
x = x/10 # integer arithmetic. 25/10 -> 2
x = x**3 # power symbol denoted by '**' instead of '^'
print x

y = 19.2
print type(y)

z = True
print type(z)

c = 2 + 1j
print type(c)

# to dynamically check object type - isinstance()
res = isinstance(c, int)
print res
res = isinstance(x, int)
print res

# casting to int
c1 = int(y) # allowed to cast float to int
c1
c2 = int(c) # cannot convert complex to int


a = "abcdef"
print type(a)

# object-oriented approach unlike R doing like captialize(a)
a.capitalize()
a.upper()
x = 100
x.bit_length()

abc_xyz = 1234
print abc_xyz

# invalid variable name
# abc.xyz =123
# print abc.xyz
