# 1. NumPy

NumPy is a well known general-purpose array-processing package. Used for large multi-dimensional array and matrix processing, with the help of a large collection of high-level mathematical functions.NumPy is very useful for handling linear algebra, Fourier transforms, and random numbers.

**Import Syntax**
```py
import numpy as np

#Creating an array object
array = np.array([
    [1, 2, 3, 5, 6],
    [2, 1, 5, 6, 7]
])
```

**Important  Function**
```py
#Print array dimension using array.ndim
print("No. of dimensions of the array: ", array.ndim, end="\n")

#Print array size using array.size
print("Size of the array: ", array.size, end="\n")

#Print array shape
print("Shape of the array: ", array.shape)
```
Output : 
```
No. of dimensions of the array: 2
Size of the array: 10
Shape of the array: (2, 5) 
```

### **Creating NumPy Arrays**

Arrays in NumPy can be created in multiple ways, with various number of Ranks, defining the size of the Array.

```py
#Creating a rank 1 array by passing one python list
list = [1, 2, 3, 5, 6]
array = np.array(list)
print(array)
```
Output :
```
[1 2 3 5 6]
```
```py
#Creating a rank 2 array by passing two python lists
list1 = [1, 2, 3, 5, 6]
list2 = [3, 1, 4, 5, 1]
array = np.array(
    [list1, list2]
)
print(array)
```
Output :
```
[[1 2 3 5 6]
[3 1 4 5 1]]
```
```py
#Creating a constant value array of complex type using np.full( )
array = np.full((2, 2), 4, dtype='complex')
print(array)
```
Output :
```
[[4.+0.j 4.+0.j ]
[4.+0.j 4.+0.j ]]
```
### **Basic slicing and indexing in a multidimensional array**
Slicing and indexing in a multidimensional array are a little bit tricky compared to slicing and indexing in a one-dimensional array.
```py
array = np.array([
    [2, 4, 5, 6],
    [3, 1, 6, 9],
    [4, 5, 1, 9],
    [2, 9, 1, 7]
])

# Slicing and indexing in 4x4 array
# Print first two rows and first two columns
print("\n", array[0:2, 0:2])

# Print all rows and last two columns
print("\n", array[:, 2:4])

# Print all column but middle two rows
print("\n", array[1:3, :])
```
Output:
```
[[2 4]
[3 1]]

[[5 6]
[6 9]
[1 9]
[1 7]]

[[3 1 6 9]
[4 5 1 9]]
```
### **NumPy Operations on Array**

In NumPy, arrays allow various operations that can be performed on a particular array or a combination of Arrays.These operations  include some basic Mathematical operations as well as Unary and Binary operations.

Examples :

```py
array = np.array([
    [2, 4, 5],
    [3, 1, 6]
])

#Adding 5 to every element in array1
newArray = array + 5
print(newArray)
```
Output :
```
[[ 7 9 10]
[ 8 6 11]]
```
```py
#Multiplying 2 to every element in array2
newArray = array * 2
print(newArray)
```
Output:
```
[[ 4 8 10]
[ 6 2 12]]
```
## References
- [More on](https://numpy.org/doc/)