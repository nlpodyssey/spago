// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

// The Matrix interface defines set and get methods to access its elements plus a few variants to perform linear algebra
// operations with other matrices, such as element-wise addition, subtraction, product and matrix-matrix multiplication.
type Matrix interface {
	// ZerosLike returns a new Dense with of the same dimensions of the receiver, initialized with zeros.
	ZerosLike() Matrix
	// OnesLike returns a new Dense with of the same dimensions of the receiver, initialized to ones.
	OnesLike() Matrix
	// Clone returns a new matrix copying the values of the receiver.
	Clone() Matrix
	// Copy copies the data to the receiver.
	Copy(other Matrix)
	// Zeros set all the values to zeros.
	Zeros()
	// Dims returns the number of rows and columns.
	Dims() (r, c int)
	// Rows returns the number of rows.
	Rows() int
	// Columns returns the number of columns.
	Columns() int
	// Size returns the size of the matrix (rows * cols).
	Size() int
	// LastIndex returns the last index respect to linear indexing.
	LastIndex() int
	// Data returns the underlying data.
	Data() []float64
	// IsVectors returns whether the matrix has one row or one column, or not.
	IsVector() bool
	// IsScalar returns whether the matrix contains a scalar, or not.
	IsScalar() bool
	// Scalar returns the scalar. It panics if the matrix contains more elements.
	Scalar() float64
	// Set sets the value v at row i and column j.
	Set(i int, j int, v float64)
	// At returns the value at row i and column j.
	At(i int, j int) float64
	// SetVec sets the value v at position i of a vector. It panics if not IsVector().
	SetVec(i int, v float64)
	// AtVec returns the value at position i of a vector. It panics if not IsVector().
	AtVec(i int) float64
	// T returns the transpose of the matrix.
	T() Matrix
	// Reshape returns a copy of the matrix. It panics if the dimensions are not compatible.
	Reshape(r, c int) Matrix
	// Apply execute the unary function fn.
	Apply(fn func(i, j int, v float64) float64, a Matrix)
	// ApplyWithAlpha executes the unary function fn, taking additional parameters alpha.
	ApplyWithAlpha(fn func(i, j int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64)
	// AddScalar performs an addition between the Matrix and a float.
	AddScalar(n float64) Matrix
	// AddScalarInPlace adds the scalar to the receiver.
	AddScalarInPlace(n float64) Matrix
	// SubScalar performs a subtraction between the Matrix and a float.
	SubScalar(n float64) Matrix
	// SubScalarInPlace subtracts the scalar to the receiver.
	SubScalarInPlace(n float64) Matrix
	// ProdScalar returns the multiplication of the float with the receiver.
	ProdScalar(n float64) Matrix
	// ProdScalarInPlace multiply a float with the receiver in place.
	ProdScalarInPlace(n float64) Matrix
	// ProdMatrixScalarInPlace multiply a matrix with a float, storing the result in the receiver.
	ProdMatrixScalarInPlace(m Matrix, n float64) Matrix
	// Add returns the addition with a matrix with the receiver.
	Add(other Matrix) Matrix
	// AddInPlace performs the addition with the other matrix in place.
	AddInPlace(other Matrix) Matrix
	// Sub returns the subtraction with a matrix with the receiver.
	Sub(other Matrix) Matrix
	// SubInPlace performs the subtraction with the other matrix in place.
	SubInPlace(other Matrix) Matrix
	// Prod performs the element-wise product with the receiver.
	Prod(other Matrix) Matrix
	// ProdInPlace performs the element-wise product with the receiver in place.
	ProdInPlace(other Matrix) Matrix
	// Div returns the result of the element-wise division.
	Div(other Matrix) Matrix
	// Div performs the result of the element-wise division in place.
	DivInPlace(other Matrix) Matrix
	// Mul performs the multiplication row by column. AB = C
	// if A is an r x c Matrix, and B is j X k, c = j the resulting Matrix C will be r x k
	Mul(other Matrix) Matrix
	// DotUnitary returns the dot product of two vectors.
	DotUnitary(other Matrix) float64
	// Pow returns a new matrix applying the power v (applying the pow function) to all elements.
	Pow(power float64) Matrix
	// Norm returns the vector norm. Use pow = 2.0 for Euclidean.
	Norm(pow float64) float64
	// Sqrt returns a new matrix applying the sqrt function to all elements.
	Sqrt() Matrix
	// ClipInPlace performs the clip in place.
	ClipInPlace(min, max float64) Matrix
	// Abs returns a new matrix applying the abs function to all elements.
	Abs() Matrix
	// Sum returns the sum of all values of the matrix.
	Sum() float64
	// Max returns the max value of the matrix.
	Max() float64
	// Min returns the min value of the matrix.
	Min() float64
	// String returns the string representation of the data.
	String() string
	// SetData sets the data.
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
