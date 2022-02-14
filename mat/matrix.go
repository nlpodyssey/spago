// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

// The Matrix interface defines set and get methods to access its elements,
// plus a few variants to perform linear algebra operations with other matrices,
// such as element-wise addition, subtraction, product and matrix-matrix
// multiplication.
type Matrix[T DType] interface {
	// Rows returns the number of rows of the matrix.
	Rows() int
	// Columns returns the number of columns of the matrix.
	Columns() int
	// Dims returns the number of rows and columns of the matrix.
	Dims() (r, c int)
	// The Size of the matrix (rows*columns).
	Size() int
	// Data returns the underlying data of the matrix, as a raw one-dimensional
	// slice of values in row-major order.
	Data() []T
	// SetData sets the content of the matrix, copying the given raw
	// data representation as one-dimensional slice.
	SetData(data []T)
	// ZerosLike returns a new matrix with the same dimensions of the
	// receiver, initialized with zeroes.
	ZerosLike() Matrix[T]
	// OnesLike returns a new matrix with the same dimensions of the
	// receiver, initialized with ones.
	OnesLike() Matrix[T]
	// Scalar returns the scalar value.
	// It panics if the matrix does not contain exactly one element.
	Scalar() T
	// Zeros sets all the values of the matrix to zero.
	Zeros()
	// Set sets the value v at row r and column c.
	// It panics if the given indices are out of range.
	Set(r int, c int, v T)
	// At returns the value at row r and column c.
	// It panics if the given indices are out of range.
	At(r int, c int) T
	// SetVec sets the value v at position i of a vector.
	// It panics if the receiver is not a vector or the position is out of range.
	SetVec(i int, v T)
	// AtVec returns the value at position i of a vector.
	// It panics if the receiver is not a vector or the position is out of range.
	AtVec(i int) T
	// ExtractRow returns a copy of the i-th row of the matrix.
	ExtractRow(i int) Matrix[T]
	// ExtractColumn returns a copy of the i-th column of the matrix.
	ExtractColumn(i int) Matrix[T]
	// View returns a new Matrix sharing the same underlying data.
	View(rows, cols int) Matrix[T]
	// Reshape returns a copy of the matrix.
	// It panics if the dimensions are incompatible.
	Reshape(r, c int) Matrix[T]
	// ReshapeInPlace changes the dimensions of the matrix in place and returns the
	// matrix itself.
	// It panics if the dimensions are incompatible.
	ReshapeInPlace(r, c int) Matrix[T]
	// ResizeVector returns a resized copy of the vector.
	//
	// If the new size is smaller than the input vector, the remaining tail
	// elements are removed. If it's bigger, the additional tail elements
	// are set to zero.
	ResizeVector(newSize int) Matrix[T]
	// T returns the transpose of the matrix.
	T() Matrix[T]
	// Add returns the addition between the receiver and another matrix.
	Add(other Matrix[T]) Matrix[T]
	// AddInPlace performs the in-place addition with the other matrix.
	AddInPlace(other Matrix[T]) Matrix[T]
	// AddScalar performs the addition between the matrix and the given value.
	AddScalar(n T) Matrix[T]
	// AddScalarInPlace adds the scalar to all values of the matrix.
	AddScalarInPlace(n T) Matrix[T]
	// Sub returns the subtraction of the other matrix from the receiver.
	Sub(other Matrix[T]) Matrix[T]
	// SubInPlace performs the in-place subtraction with the other matrix.
	SubInPlace(other Matrix[T]) Matrix[T]
	// SubScalar performs a subtraction between the matrix and the given value.
	SubScalar(n T) Matrix[T]
	// SubScalarInPlace subtracts the scalar from the receiver's values.
	SubScalarInPlace(n T) Matrix[T]
	// Prod performs the element-wise product between the receiver and the other matrix.
	Prod(other Matrix[T]) Matrix[T]
	// ProdInPlace performs the in-place element-wise product with the other matrix.
	ProdInPlace(other Matrix[T]) Matrix[T]
	// ProdScalar returns the multiplication between the matrix and the given value.
	ProdScalar(n T) Matrix[T]
	// ProdScalarInPlace performs the in-place multiplication between the
	// matrix and the given value.
	ProdScalarInPlace(n T) Matrix[T]
	// ProdMatrixScalarInPlace multiplies the given matrix with the value,
	// storing the result in the receiver.
	ProdMatrixScalarInPlace(m Matrix[T], n T) Matrix[T]
	// Div returns the result of the element-wise division of the receiver by the other matrix.
	Div(other Matrix[T]) Matrix[T]
	// DivInPlace performs the in-place element-wise division of the receiver by the other matrix.
	DivInPlace(other Matrix[T]) Matrix[T]
	// Mul performs the multiplication row by column.
	// If A is an i×j Matrix, and B is j×k, then the resulting Matrix
	// C = AB will be i×k.
	Mul(other Matrix[T]) Matrix[T]
	// MulT performs the matrix multiplication row by column.
	// ATB = C, where AT is the transpose of B
	// if A is an r x c Matrix, and B is j x k, r = j the resulting
	// Matrix C will be c x k.
	MulT(other Matrix[T]) Matrix[T]
	// DotUnitary returns the dot product of two vectors.
	DotUnitary(other Matrix[T]) T
	// ClipInPlace clips in place each value of the matrix.
	ClipInPlace(min, max T) Matrix[T]
	// Maximum returns a new matrix containing the element-wise maxima.
	Maximum(other Matrix[T]) Matrix[T]
	// Minimum returns a new matrix containing the element-wise minima.
	Minimum(other Matrix[T]) Matrix[T]
	// Abs returns a new matrix applying the absolute value function to all elements.
	Abs() Matrix[T]
	// Pow returns a new matrix, applying the power function with given exponent
	// to all elements of the matrix.
	Pow(power T) Matrix[T]
	// Sqrt returns a new matrix applying the square root function to all elements.
	Sqrt() Matrix[T]
	// Sum returns the sum of all values of the matrix.
	Sum() T
	// Max returns the maximum value of the matrix.
	Max() T
	// Min returns the minimum value of the matrix.
	Min() T
	// Range creates a new vector initialized with data extracted from the
	// matrix raw data, from start (inclusive) to end (exclusive).
	Range(start, end int) Matrix[T]
	// SplitV extract N vectors from the Matrix.
	// N[i] has size sizes[i].
	SplitV(sizes ...int) []Matrix[T]
	// Augment places the identity matrix at the end of the original matrix.
	Augment() Matrix[T]
	// SwapInPlace swaps two rows of the matrix in place.
	SwapInPlace(r1, r2 int) Matrix[T]
	// PadRows returns a copy of the matrix with n additional tail rows.
	// The additional elements are set to zero.
	PadRows(n int) Matrix[T]
	// PadColumns returns a copy of the matrix with n additional tail columns.
	// The additional elements are set to zero.
	PadColumns(n int) Matrix[T]
	// Norm returns the vector's norm. Use pow = 2.0 to compute the Euclidean norm.
	Norm(pow T) T
	// Pivoting returns the partial pivots of a square matrix to reorder rows.
	// Considerate square sub-matrix from element (offset, offset).
	Pivoting(row int) (Matrix[T], bool, []int)
	// Normalize2 normalizes an array with the Euclidean norm.
	Normalize2() Matrix[T]
	// LU performs lower–upper (LU) decomposition of a square matrix D such as
	// PLU = D, L is lower diagonal and U is upper diagonal, p are pivots.
	LU() (l, u, p Matrix[T])
	// Inverse returns the inverse of the Matrix.
	Inverse() Matrix[T]
	// Apply creates a new matrix executing the unary function fn.
	Apply(fn func(r, c int, v T) T) Matrix[T]
	// ApplyInPlace executes the unary function fn.
	ApplyInPlace(fn func(r, c int, v T) T, a Matrix[T])
	// ApplyWithAlpha creates a new matrix executing the unary function fn,
	// taking additional parameters alpha.
	ApplyWithAlpha(fn func(r, c int, v T, alpha ...T) T, alpha ...T) Matrix[T]
	// ApplyWithAlphaInPlace executes the unary function fn, taking additional parameters alpha.
	ApplyWithAlphaInPlace(fn func(r, c int, v T, alpha ...T) T, a Matrix[T], alpha ...T)
	// DoNonZero calls a function for each non-zero element of the matrix.
	// The parameters of the function are the element's indices and value.
	DoNonZero(fn func(r, c int, v T))
	// DoVecNonZero calls a function for each non-zero element of the vector.
	// The parameters of the function are the element's index and value.
	DoVecNonZero(fn func(i int, v T))
	// Clone returns a new matrix, copying all its values from the receiver.
	Clone() Matrix[T]
	// Copy copies the data from the other matrix to the receiver.
	Copy(other Matrix[T])
	// String returns a string representation of the matrix.
	String() string
}

// IsVector returns whether the matrix is either a row or column vector
// (dimensions N×1 or 1×N).
func IsVector[T DType](m Matrix[T]) bool {
	return m.Rows() == 1 || m.Columns() == 1
}

// IsScalar returns whether the matrix contains exactly one scalar value
// (dimensions 1×1).
func IsScalar[T DType](m Matrix[T]) bool {
	return m.Size() == 1
}

// SameDims reports whether the two matrices have the same dimensions.
func SameDims[T DType](a, b Matrix[T]) bool {
	return a.Rows() == b.Rows() && a.Columns() == b.Columns()
}

// VectorsOfSameSize reports whether both matrices are vectors (indifferently
// row or column vectors) and have the same size.
func VectorsOfSameSize[T DType](a, b Matrix[T]) bool {
	return a.Size() == b.Size() && IsVector(a) && IsVector(b)
}

// ConcatV returns a new Matrix created concatenating the input matrices vertically.
func ConcatV[T DType](vs ...Matrix[T]) *Dense[T] {
	cup := 0
	for _, v := range vs {
		cup += v.Size()
	}
	data := make([]T, 0, cup)
	for _, v := range vs {
		if !IsVector(v) {
			panic("mat: required vector, found matrix")
		}
		data = append(data, v.Data()...)
	}
	return NewVecDense(data)
}

// ConcatH returns a new Matrix created concatenating the input matrices horizontally.
func ConcatH[T DType](ms ...Matrix[T]) *Dense[T] {
	rows := len(ms)
	cols := ms[0].Rows()
	out := NewEmptyDense[T](rows, cols)
	for i, x := range ms {
		for j := 0; j < cols; j++ {
			out.Set(i, j, x.At(j, 0))
		}
	}
	return out
}

// Stack returns a new Matrix created concatenating the input vectors horizontally.
func Stack[T DType](vs ...Matrix[T]) *Dense[T] {
	rows := len(vs)
	cols := vs[0].Size()
	out := densePool[T]().Get(rows, cols) // it doesn't need to be empty, because we are going to fill it up again
	start := 0
	end := cols
	for _, v := range vs {
		copy(out.data[start:end], v.Data())
		start = end
		end += cols
	}
	return out
}
