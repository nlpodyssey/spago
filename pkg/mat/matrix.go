// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

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
	Data() []float64
	// IsVector returns whether the matrix is either a row or column vector.
	IsVector() bool
	// IsScalar returns whether the matrix contains exactly one scalar value.
	IsScalar() bool
	// Scalar returns the scalar value.
	// It panics if the matrix does not contain exactly one element.
	Scalar() float64
	// Set sets the value v at row i and column j.
	Set(i int, j int, v float64)
	// At returns the value at row i and column j.
	At(i int, j int) float64
	// SetVec sets the value v at position i of a vector.
	// It panics if the receiver is not a vector.
	SetVec(i int, v float64)
	// AtVec returns the value at position i of a vector.
	// It panics if the receiver is not a vector.
	AtVec(i int) float64
	// T returns the transpose of the matrix.
	T() Matrix
	// Reshape returns a copy of the matrix.
	// It panics if the dimensions are incompatible.
	Reshape(r, c int) Matrix
	// Apply executes the unary function fn.
	Apply(fn func(i, j int, v float64) float64, a Matrix)
	// ApplyWithAlpha executes the unary function fn, taking additional parameters alpha.
	ApplyWithAlpha(fn func(i, j int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64)
	// AddScalar performs the addition between the matrix and the given value.
	AddScalar(n float64) Matrix
	// AddScalarInPlace adds the scalar to all values of the matrix.
	AddScalarInPlace(n float64) Matrix
	// SubScalar performs a subtraction between the matrix and the given value.
	SubScalar(n float64) Matrix
	// SubScalarInPlace subtracts the scalar from the receiver's values.
	SubScalarInPlace(n float64) Matrix
	// ProdScalar returns the multiplication between the matrix and the given value.
	ProdScalar(n float64) Matrix
	// ProdScalarInPlace performs the in-place multiplication between the matrix and
	// the given value.
	ProdScalarInPlace(n float64) Matrix
	// ProdMatrixScalarInPlace multiplies the given matrix with the value, storing the
	// result in the receiver.
	ProdMatrixScalarInPlace(m Matrix, n float64) Matrix
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
	DotUnitary(other Matrix) float64
	// Pow returns a new matrix, applying the power function with given exponent to all elements
	// of the matrix.
	Pow(power float64) Matrix
	// Norm returns the vector's norm. Use pow = 2.0 to compute the Euclidean norm.
	Norm(pow float64) float64
	// Sqrt returns a new matrix applying the square root function to all elements.
	Sqrt() Matrix
	// ClipInPlace clips in place each value of the matrix.
	ClipInPlace(min, max float64) Matrix
	// Abs returns a new matrix applying the absolute value function to all elements.
	Abs() Matrix
	// Sum returns the sum of all values of the matrix.
	Sum() float64
	// Max returns the maximum value of the matrix.
	Max() float64
	// Min returns the minimum value of the matrix.
	Min() float64
	// String returns a string representation of the matrix data.
	String() string
	// SetData sets the values of the matrix, given a raw one-dimensional slice
	// data representation.
	SetData(data []float64)
}

// ConcatV returns a new Matrix created concatenating the input matrices vertically.
func ConcatV(vs ...Matrix) Matrix {
	cup := 0
	for _, v := range vs {
		cup += v.Size()
	}
	data := make([]float64, 0, cup)
	for _, v := range vs {
		if !v.IsVector() {
			panic("mat: required vector, found matrix")
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

// Stack ...
func Stack(vs ...*Dense) *Dense {
	rows := len(vs)
	cols := vs[0].size
	out := GetDenseWorkspace(rows, cols) // it doesn't need to be empty, because we are going to fill it up again
	start := 0
	end := cols
	for _, v := range vs {
		copy(out.data[start:end], v.data)
		start = end
		end += cols
	}
	return out
}
