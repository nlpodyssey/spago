# SkipNumbers Dataset
The training and test data-sets contain 100,000 and 10,000 examples respectively.
Each example is encoded in a JSON array of length 21. The first 20 numbers are input sequence and the last one is the label.  

Each input sequence has a length *T* = 20 with L positive integers *x[0:T−1]*. It is formed by randomly sampling 20 numbers from the integer set *{0, 1, ..., 9}*, and setting *x[x[T-1]]* as the label of each example.

## Example

Given the record

```
[8, 5, 1, 7, 4, 3, 7]
```

the inputs sequence is 

```
[8, 5, 1, 7, 4, 3]
```

and the label is

```
7
```

because of *x[x[T-1]]*:

```
x[3] = 7
```
