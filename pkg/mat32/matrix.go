// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

// The Matrix interface defines set and get methods to access its elements plus a few variants to perform linear algebra
// operations with other matrices, such as element-wise addition, subtraction, product and matrix-matrix multiplication.
type Matrix interface {
	// ZerosLike returns a new matrix with the same dimensions of the receiver,
	// initialized with zeroes.
	ZerosLike() Matrix
	// OnesLike returns a new matrix with the same dimensions of the receiver,
	// initialized with ones.
	OnesLike() Matrix
	// Clone returns a new matrix, copying all its values from the receiver.
	Clone() Matrix
	// Copy copies the data from the other matrix to the receiver.
	Copy(other Matrix)
	// Zeros sets all the values of the matrix to zero.
	Zeros()
	// Dims returns the number of rows and columns of the matrix.
	Dims() (r, c int)
	// Rows returns the number of rows of the matrix.
	Rows() int
	// Columns returns the number of columns of the matrix.
	Columns() int
	// Size returns the size of the matrix (rows × columns).
	Size() int
	// LastIndex returns the last element's index, in respect of linear indexing.
	// It returns -1 if the matrix is empty.
	LastIndex() int
	// Data returns the underlying data of the matrix, as a raw one-dimensional slice of values.
	Data() []Float
	// IsVector returns whether the matrix is either a row or column vector.
	IsVector() bool
	// IsScalar returns whether the matrix contains exactly one scalar value.
	IsScalar() bool
	// Scalar returns the scalar value.
	// It panics if the matrix does not contain exactly one element.
	Scalar() Float
	// Set sets the value v at row i and column j.
	Set(i int, j int, v Float)
	// At returns the value at row i and column j.
	At(i int, j int) Float
	// SetVec sets the value v at position i of a vector.
	// It panics if the receiver is not a vector.
	SetVec(i int, v Float)
	// AtVec returns the value at position i of a vector.
	// It panics if the receiver is not a vector.
	AtVec(i int) Float
	// T returns the transpose of the matrix.
	T() Matrix
	// Reshape returns a copy of the matrix.
	// It panics if the dimensions are incompatible.
	Reshape(r, c int) Matrix
	// Apply executes the unary function fn.
	Apply(fn func(i, j int, v Float) Float, a Matrix)
	// ApplyWithAlpha executes the unary function fn, taking additional parameters alpha.
	ApplyWithAlpha(fn func(i, j int, v Float, alpha ...Float) Float, a Matrix, alpha ...Float)
	// AddScalar performs the addition between the matrix and the given value.
	AddScalar(n Float) Matrix
	// AddScalarInPlace adds the scalar to all values of the matrix.
	AddScalarInPlace(n Float) Matrix
	// SubScalar performs a subtraction between the matrix and the given value.
	SubScalar(n Float) Matrix
	// SubScalarInPlace subtracts the scalar from the receiver's values.
	SubScalarInPlace(n Float) Matrix
	// ProdScalar returns the multiplication between the matrix and the given value.
	ProdScalar(n Float) Matrix
	// ProdScalarInPlace performs the in-place multiplication between the matrix and
	// the given value.
	ProdScalarInPlace(n Float) Matrix
	// ProdMatrixScalarInPlace multiplies the given matrix with the value, storing the
	// result in the receiver.
	ProdMatrixScalarInPlace(m Matrix, n Float) Matrix
	// Add returns the addition between the receiver and another matrix.
	Add(other Matrix) Matrix
	// AddInPlace performs the in-place addition with the other matrix.
	AddInPlace(other Matrix) Matrix
	// Sub returns the subtraction of the other matrix from the receiver.
	Sub(other Matrix) Matrix
	// SubInPlace performs the in-place subtraction with the other matrix.
	SubInPlace(other Matrix) Matrix
	// Prod performs the element-wise product between the receiver and the other matrix.
	Prod(other Matrix) Matrix
	// ProdInPlace performs the in-place element-wise product with the other matrix.
	ProdInPlace(other Matrix) Matrix
	// Div returns the result of the element-wise division of the receiver by the other matrix.
	Div(other Matrix) Matrix
	// DivInPlace performs the in-place element-wise division of the receiver by the other matrix.
	DivInPlace(other Matrix) Matrix
	// Mul performs the multiplication row by column.
	// If A is an i×j Matrix, and B is j×k, then the resulting Matrix C = AB will be i×k.
	Mul(other Matrix) Matrix
	// DotUnitary returns the dot product of two vectors.
	DotUnitary(other Matrix) Float
	// Pow returns a new matrix, applying the power function with given exponent to all elements
	// of the matrix.
	Pow(power Float) Matrix
	// Norm returns the vector's norm. Use pow = 2.0 to compute the Euclidean norm.
	Norm(pow Float) Float
	// Sqrt returns a new matrix applying the square root function to all elements.
	Sqrt() Matrix
	// ClipInPlace clips in place each value of the matrix.
	ClipInPlace(min, max Float) Matrix
	// SplitV extract N vectors from the Matrix.
	// N[i] has size sizes[i].
	SplitV(sizes ...int) []Matrix
	// Minimum returns a new matrix containing the element-wise minima.
	Minimum(other Matrix) Matrix
	// Maximum returns a new matrix containing the element-wise maxima.
	Maximum(other Matrix) Matrix
	// MulT performs the matrix multiplication row by column. ATB = C, where AT is the transpose of B
	// if A is an r x c Matrix, and B is j x k, r = j the resulting Matrix C will be c x k.
	MulT(other Matrix) Matrix
	// Inverse returns the inverse of the Matrix.
	Inverse() Matrix
	// DoNonZero calls a function for each non-zero element of the matrix.
	// The parameters of the function are the element indices and its value.
	DoNonZero(fn func(i, j int, v Float))
	// Abs returns a new matrix applying the absolute value function to all elements.
	Abs() Matrix
	// Sum returns the sum of all values of the matrix.
	Sum() Float
	// Max returns the maximum value of the matrix.
	Max() Float
	// Min returns the minimum value of the matrix.
	Min() Float
	// String returns a string representation of the matrix data.
	String() string
	// SetData sets the values of the matrix, given a raw one-dimensional slice
	// data representation.
	SetData(data []Float)
}

// ConcatV returns a new Matrix created concatenating the input matrices vertically.
func ConcatV(vs ...Matrix) Matrix {
	cup := 0
	for _, v := range vs {
		cup += v.Size()
	}
	data := make([]Float, 0, cup)
	for _, v := range vs {
		if !v.IsVector() {
			panic("mat32: required vector, found matrix")
		}
		data = append(data, v.Data()...)
	}
	return NewVecDense(data)
}

// ConcatH returns a new Matrix created concatenating the input matrices horizontally.
func ConcatH(ms ...Matrix) *Dense {
	rows := len(ms)
	cols := ms[0].Rows()
	out := NewEmptyDense(rows, cols)
	for i, x := range ms {
		for j := 0; j < cols; j++ {
			out.Set(i, j, x.At(j, 0))
		}
	}
	return out
}

// Stack returns a new Matrix created concatenating the input vectors horizontally.
func Stack(vs ...Matrix) Matrix {
	rows := len(vs)
	cols := vs[0].Size()
	out := GetDenseWorkspace(rows, cols) // it doesn't need to be empty, because we are going to fill it up again
	start := 0
	end := cols
	for _, v := range vs {
		copy(out.data[start:end], v.Data())
		start = end
		end += cols
	}
	return out
}
