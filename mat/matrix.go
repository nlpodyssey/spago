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
	Data() FloatSliceInterface
	// SetData sets the content of the matrix, copying the given raw
	// data representation as one-dimensional slice.
	SetData(data FloatSliceInterface)
	// ZerosLike returns a new matrix with the same dimensions of the
	// receiver, initialized with zeroes.
	ZerosLike() Matrix[T]
	// OnesLike returns a new matrix with the same dimensions of the
	// receiver, initialized with ones.
	OnesLike() Matrix[T]
	// Scalar returns the scalar value.
	// It panics if the matrix does not contain exactly one element.
	Scalar() FloatInterface
	// Zeros sets all the values of the matrix to zero.
	Zeros()
	// Set sets the scalar value from a 1×1 matrix at row r and column c.
	// It panics if the given matrix is not 1×1, or if indices are out of range.
	Set(r int, c int, m Matrix[T])
	// At returns the value at row r and column c as a 1×1 matrix.
	// It panics if the given indices are out of range.
	At(r int, c int) Matrix[T]
	// SetScalar sets the value v at row r and column c.
	// It panics if the given indices are out of range.
	SetScalar(r int, c int, v FloatInterface)
	// ScalarAt returns the value at row r and column c.
	// It panics if the given indices are out of range.
	ScalarAt(r int, c int) FloatInterface
	// SetVec sets the scalar value from a 1×1 matrix at position i of a
	// vector. It panics if the receiver is not a vector, or the given matrix is
	// not 1×1, or the position is out of range.
	SetVec(i int, m Matrix[T])
	// AtVec returns the value at position i of a vector as a 1×1 matrix.
	// It panics if the receiver is not a vector or the position is out of range.
	AtVec(i int) Matrix[T]
	// SetVecScalar sets the value v at position i of a vector.
	// It panics if the receiver is not a vector or the position is out of range.
	SetVecScalar(i int, v FloatInterface)
	// ScalarAtVec returns the value at position i of a vector.
	// It panics if the receiver is not a vector or the position is out of range.
	ScalarAtVec(i int) FloatInterface
	// ExtractRow returns a copy of the i-th row of the matrix,
	// as a row vector (1×cols).
	ExtractRow(i int) Matrix[T]
	// ExtractColumn returns a copy of the i-th column of the matrix,
	// as a column vector (rows×1).
	ExtractColumn(i int) Matrix[T]
	// View returns a new Matrix sharing the same underlying data.
	View(rows, cols int) Matrix[T]
	// Slice returns a new matrix obtained by slicing the receiver across the
	// given positions. The parameters "fromRow" and "fromCol" are inclusive,
	// while "toRow" and "toCol" are exclusive.
	Slice(fromRow, fromCol, toRow, toCol int) Matrix[T]
	// Reshape returns a copy of the matrix.
	// It panics if the dimensions are incompatible.
	Reshape(r, c int) Matrix[T]
	// ReshapeInPlace changes the dimensions of the matrix in place and returns the
	// matrix itself.
	// It panics if the dimensions are incompatible.
	ReshapeInPlace(r, c int) Matrix[T]
	// Flatten creates a new row vector (1×size) corresponding to the
	// "flattened" row-major ordered representation of the initial matrix.
	Flatten() Matrix[T]
	// FlattenInPlace transforms the matrix in place, changing its dimensions,
	// obtaining a row vector (1×size) containing the "flattened" row-major
	// ordered representation of the initial value.
	// It returns the matrix itself.
	FlattenInPlace() Matrix[T]
	// ResizeVector returns a resized copy of the vector.
	//
	// If the new size is smaller than the input vector, the remaining tail
	// elements are removed. If it's bigger, the additional tail elements
	// are set to zero.
	ResizeVector(newSize int) Matrix[T]
	// T returns the transpose of the matrix.
	T() Matrix[T]
	// TransposeInPlace transposes the matrix in place, and returns the
	// matrix itself.
	TransposeInPlace() Matrix[T]
	// Add returns the addition between the receiver and another matrix.
	Add(other Matrix[T]) Matrix[T]
	// AddInPlace performs the in-place addition with the other matrix.
	AddInPlace(other Matrix[T]) Matrix[T]
	// AddScalar performs the addition between the matrix and the given value.
	AddScalar(n float64) Matrix[T]
	// AddScalarInPlace adds the scalar to all values of the matrix.
	AddScalarInPlace(n float64) Matrix[T]
	// Sub returns the subtraction of the other matrix from the receiver.
	Sub(other Matrix[T]) Matrix[T]
	// SubInPlace performs the in-place subtraction with the other matrix.
	SubInPlace(other Matrix[T]) Matrix[T]
	// SubScalar performs a subtraction between the matrix and the given value.
	SubScalar(n float64) Matrix[T]
	// SubScalarInPlace subtracts the scalar from the receiver's values.
	SubScalarInPlace(n float64) Matrix[T]
	// Prod performs the element-wise product between the receiver and the other matrix.
	Prod(other Matrix[T]) Matrix[T]
	// ProdInPlace performs the in-place element-wise product with the other matrix.
	ProdInPlace(other Matrix[T]) Matrix[T]
	// ProdScalar returns the multiplication between the matrix and the given value.
	ProdScalar(n float64) Matrix[T]
	// ProdScalarInPlace performs the in-place multiplication between the
	// matrix and the given value.
	ProdScalarInPlace(n float64) Matrix[T]
	// ProdMatrixScalarInPlace multiplies the given matrix with the value,
	// storing the result in the receiver.
	ProdMatrixScalarInPlace(m Matrix[T], n float64) Matrix[T]
	// Div returns the result of the element-wise division of the receiver by the other matrix.
	Div(other Matrix[T]) Matrix[T]
	// DivInPlace performs the in-place element-wise division of the receiver by the other matrix.
	DivInPlace(other Matrix[T]) Matrix[T]
	// Mul performs the multiplication row by column.
	// If A is an i×j Matrix, and B is j×k, then the resulting Matrix
	// C = AB will be i×k.
	Mul(other Matrix[T]) Matrix[T]
	// MulT performs the matrix multiplication row by column.
	// ATB = C, where AT is the transpose of A
	// if A is an r x c Matrix, and B is j x k, r = j the resulting
	// Matrix C will be c x k.
	MulT(other Matrix[T]) Matrix[T]
	// DotUnitary returns the dot product of two vectors as a scalar Matrix.
	DotUnitary(other Matrix[T]) Matrix[T]
	// ClipInPlace clips in place each value of the matrix.
	ClipInPlace(min, max float64) Matrix[T]
	// Maximum returns a new matrix containing the element-wise maxima.
	Maximum(other Matrix[T]) Matrix[T]
	// Minimum returns a new matrix containing the element-wise minima.
	Minimum(other Matrix[T]) Matrix[T]
	// Abs returns a new matrix applying the absolute value function to all elements.
	Abs() Matrix[T]
	// Pow returns a new matrix, applying the power function with given exponent
	// to all elements of the matrix.
	Pow(power float64) Matrix[T]
	// Sqrt returns a new matrix applying the square root function to all elements.
	Sqrt() Matrix[T]
	// Sum returns the sum of all values of the matrix as a scalar Matrix.
	Sum() Matrix[T]
	// Max returns the maximum value of the matrix as a scalar Matrix.
	Max() Matrix[T]
	// Min returns the minimum value of the matrix as a scalar Matrix.
	Min() Matrix[T]
	// ArgMax returns the index of the vector's element with the maximum value.
	ArgMax() int
	// Softmax applies the softmax function to the vector, returning the
	// result as a new column vector.
	Softmax() Matrix[T]
	// CumSum computes the cumulative sum of the vector's elements, returning
	// the result as a new column vector.
	CumSum() Matrix[T]
	// Range creates a new vector initialized with data extracted from the
	// matrix raw data, from start (inclusive) to end (exclusive).
	Range(start, end int) Matrix[T]
	// SplitV splits the vector in N chunks of given sizes,
	// so that N[i] has size sizes[i].
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
	// AppendRows returns a copy of the matrix with len(vs) additional tail rows,
	// being each new row filled with the values of each given vector.
	//
	// It accepts row or column vectors indifferently, virtually treating all of
	// them as row vectors.
	AppendRows(vs ...Matrix[T]) Matrix[T]
	// Norm returns the vector's norm. Use pow = 2.0 to compute the Euclidean norm.
	// The result is a scalar Matrix.
	Norm(pow float64) Matrix[T]
	// Pivoting returns the partial pivots of a square matrix to reorder rows.
	// Considerate square sub-matrix from element (offset, offset).
	Pivoting(row int) (Matrix[T], bool, [2]int)
	// Normalize2 normalizes an array with the Euclidean norm.
	Normalize2() Matrix[T]
	// LU performs lower–upper (LU) decomposition of a square matrix D such as
	// PLU = D, L is lower diagonal and U is upper diagonal, p are pivots.
	LU() (l, u, p Matrix[T])
	// Inverse returns the inverse of the Matrix.
	Inverse() Matrix[T]
	// Apply creates a new matrix executing the unary function fn.
	Apply(fn func(r, c int, v float64) float64) Matrix[T]
	// ApplyInPlace executes the unary function fn over the matrix a,
	// and stores the result in the receiver, returning the receiver itself.
	ApplyInPlace(fn func(r, c int, v float64) float64, a Matrix[T]) Matrix[T]
	// ApplyWithAlpha creates a new matrix executing the unary function fn,
	// taking additional parameters alpha.
	ApplyWithAlpha(fn func(r, c int, v float64, alpha ...float64) float64, alpha ...float64) Matrix[T]
	// ApplyWithAlphaInPlace executes the unary function fn over the matrix a,
	// taking additional parameters alpha, and stores the result in the
	// receiver, returning the receiver itself.
	ApplyWithAlphaInPlace(fn func(r, c int, v float64, alpha ...float64) float64, a Matrix[T], alpha ...float64) Matrix[T]
	// DoNonZero calls a function for each non-zero element of the matrix.
	// The parameters of the function are the element's indices and value.
	DoNonZero(fn func(r, c int, v float64))
	// DoVecNonZero calls a function for each non-zero element of the vector.
	// The parameters of the function are the element's index and value.
	DoVecNonZero(fn func(i int, v float64))
	// Clone returns a new matrix, copying all its values from the receiver.
	Clone() Matrix[T]
	// Copy copies the data from the other matrix to the receiver.
	Copy(other Matrix[T])
	// String returns a string representation of the matrix.
	String() string
}

// Data returns the underlying data of the matrix, as a raw one-dimensional
// slice of values in row-major order.
func Data[T DType](m Matrix[T]) []T {
	return DTFloatSlice[T](m.Data())
}

// SetData sets the content of the matrix, copying the given raw
// data representation as one-dimensional slice.
func SetData[T DType](m Matrix[T], data []T) {
	m.SetData(FloatSlice(data))
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

// ConcatV concatenates two or more vectors "vertically", creating a new Dense
// column vector. It accepts row or column vectors indifferently, virtually
// treating all of them as column vectors.
func ConcatV[T DType](vs ...Matrix[T]) *Dense[T] {
	size := 0
	for _, v := range vs {
		if !IsVector(v) {
			panic("mat: expected vector")
		}
		size += v.Size()
	}
	out := densePool[T]().Get(size, 1)
	data := out.data[:0] // convenient for using append below
	for _, v := range vs {
		data = append(data, Data[T](v)...)
	}
	out.data = data
	return out
}

// Stack stacks two or more vectors of the same size on top of each other,
// creating a new Dense matrix where each row contains the data of each
// input vector.
// It accepts row or column vectors indifferently, virtually treating all of
// them as row vectors.
func Stack[T DType](vs ...Matrix[T]) *Dense[T] {
	if len(vs) == 0 {
		return densePool[T]().Get(0, 0)
	}
	cols := vs[0].Size()
	out := densePool[T]().Get(len(vs), cols)
	data := out.data
	for i, v := range vs {
		if !IsVector(v) {
			panic("mat: expected vector")
		}
		if v.Size() != cols {
			panic("mat: all vectors must have the same size")
		}
		offset := i * cols
		copy(data[offset:offset+cols], Data[T](v))
	}
	return out
}

// Equal reports whether matrices a and b have the same shape and elements.
func Equal[T DType](a, b Matrix[T]) bool {
	return a.Rows() == b.Rows() &&
		a.Columns() == b.Columns() &&
		a.Data().Equals(b.Data())
}

// InDelta reports whether matrices a and b have the same shape and
// all elements at the same positions are within delta.
func InDelta[T DType](a, b Matrix[T], delta float64) bool {
	return a.Rows() == b.Rows() &&
		a.Columns() == b.Columns() &&
		a.Data().InDelta(b.Data(), delta)
}
