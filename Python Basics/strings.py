s1 = "abcdef"
s1
print type(s1)
s1[0:3] # slicing
s1[::2] # get alternate characters from the start = 'ace'
s1[0] = "z" # this does not work; does not support in-place item assignment

s2 = s1.replace("ab","xy")
s3 = s1.capitalize()
s3

isinstance(s3, str)

# convert string to list of characters
s4 = list(s1)
s4 # ['a', 'b', 'c', 'd', 'e', 'f']. See how string is converted to list
s4[0] = 'x'

# Wrong way to convert list to string. List converted to string in "raw" form.
s5 = str(s4) # Bad - "['x', 'b', 'c', 'd', 'e', 'f']". Don't use blindly
s5

# correct way of converting list to string
s6 = ''.join(map(str,s4)) # casting (to string) applied to all elements of a list
# each string 'joined'/appended to an initially empty string
type(s6)
s6
