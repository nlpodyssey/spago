// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

// The Matrix interface defines set and get methods to access its elements,
// plus a few variants to perform linear algebra operations with other matrices,
// such as element-wise addition, subtraction, product and matrix-matrix
// multiplication.
type Matrix interface {
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
	ZerosLike() Matrix
	// OnesLike returns a new matrix with the same dimensions of the
	// receiver, initialized with ones.
	OnesLike() Matrix
	// Scalar returns the scalar value.
	// It panics if the matrix does not contain exactly one element.
	Scalar() FloatInterface
	// Zeros sets all the values of the matrix to zero.
	Zeros()
	// Set sets the scalar value from a 1×1 matrix at row r and column c.
	// It panics if the given matrix is not 1×1, or if indices are out of range.
	Set(r int, c int, m Matrix)
	// At returns the value at row r and column c as a 1×1 matrix.
	// It panics if the given indices are out of range.
	At(r int, c int) Matrix
	// SetScalar sets the value v at row r and column c.
	// It panics if the given indices are out of range.
	SetScalar(r int, c int, v FloatInterface)
	// ScalarAt returns the value at row r and column c.
	// It panics if the given indices are out of range.
	ScalarAt(r int, c int) FloatInterface
	// SetVec sets the scalar value from a 1×1 matrix at position i of a
	// vector. It panics if the receiver is not a vector, or the given matrix is
	// not 1×1, or the position is out of range.
	SetVec(i int, m Matrix)
	// AtVec returns the value at position i of a vector as a 1×1 matrix.
	// It panics if the receiver is not a vector or the position is out of range.
	AtVec(i int) Matrix
	// SetVecScalar sets the value v at position i of a vector.
	// It panics if the receiver is not a vector or the position is out of range.
	SetVecScalar(i int, v FloatInterface)
	// ScalarAtVec returns the value at position i of a vector.
	// It panics if the receiver is not a vector or the position is out of range.
	ScalarAtVec(i int) FloatInterface
	// ExtractRow returns a copy of the i-th row of the matrix,
	// as a row vector (1×cols).
	ExtractRow(i int) Matrix
	// ExtractColumn returns a copy of the i-th column of the matrix,
	// as a column vector (rows×1).
	ExtractColumn(i int) Matrix
	// View returns a new Matrix sharing the same underlying data.
	View(rows, cols int) Matrix
	// Slice returns a new matrix obtained by slicing the receiver across the
	// given positions. The parameters "fromRow" and "fromCol" are inclusive,
	// while "toRow" and "toCol" are exclusive.
	Slice(fromRow, fromCol, toRow, toCol int) Matrix
	// Reshape returns a copy of the matrix.
	// It panics if the dimensions are incompatible.
	Reshape(r, c int) Matrix
	// ReshapeInPlace changes the dimensions of the matrix in place and returns the
	// matrix itself.
	// It panics if the dimensions are incompatible.
	ReshapeInPlace(r, c int) Matrix
	// Flatten creates a new row vector (1×size) corresponding to the
	// "flattened" row-major ordered representation of the initial matrix.
	Flatten() Matrix
	// FlattenInPlace transforms the matrix in place, changing its dimensions,
	// obtaining a row vector (1×size) containing the "flattened" row-major
	// ordered representation of the initial value.
	// It returns the matrix itself.
	FlattenInPlace() Matrix
	// ResizeVector returns a resized copy of the vector.
	//
	// If the new size is smaller than the input vector, the remaining tail
	// elements are removed. If it's bigger, the additional tail elements
	// are set to zero.
	ResizeVector(newSize int) Matrix
	// T returns the transpose of the matrix.
	T() Matrix
	// TransposeInPlace transposes the matrix in place, and returns the
	// matrix itself.
	TransposeInPlace() Matrix
	// Add returns the addition between the receiver and another matrix.
	Add(other Matrix) Matrix
	// AddInPlace performs the in-place addition with the other matrix.
	AddInPlace(other Matrix) Matrix
	// AddScalar performs the addition between the matrix and the given value.
	AddScalar(n float64) Matrix
	// AddScalarInPlace adds the scalar to all values of the matrix.
	AddScalarInPlace(n float64) Matrix
	// Sub returns the subtraction of the other matrix from the receiver.
	Sub(other Matrix) Matrix
	// SubInPlace performs the in-place subtraction with the other matrix.
	SubInPlace(other Matrix) Matrix
	// SubScalar performs a subtraction between the matrix and the given value.
	SubScalar(n float64) Matrix
	// SubScalarInPlace subtracts the scalar from the receiver's values.
	SubScalarInPlace(n float64) Matrix
	// Prod performs the element-wise product between the receiver and the other matrix.
	Prod(other Matrix) Matrix
	// ProdInPlace performs the in-place element-wise product with the other matrix.
	ProdInPlace(other Matrix) Matrix
	// ProdScalar returns the multiplication between the matrix and the given value.
	ProdScalar(n float64) Matrix
	// ProdScalarInPlace performs the in-place multiplication between the
	// matrix and the given value.
	ProdScalarInPlace(n float64) Matrix
	// ProdMatrixScalarInPlace multiplies the given matrix with the value,
	// storing the result in the receiver.
	ProdMatrixScalarInPlace(m Matrix, n float64) Matrix
	// Div returns the result of the element-wise division of the receiver by the other matrix.
	Div(other Matrix) Matrix
	// DivInPlace performs the in-place element-wise division of the receiver by the other matrix.
	DivInPlace(other Matrix) Matrix
	// Mul performs the multiplication row by column.
	// If A is an i×j Matrix, and B is j×k, then the resulting Matrix
	// C = AB will be i×k.
	Mul(other Matrix) Matrix
	// MulT performs the matrix multiplication row by column.
	// ATB = C, where AT is the transpose of A
	// if A is an r x c Matrix, and B is j x k, r = j the resulting
	// Matrix C will be c x k.
	MulT(other Matrix) Matrix
	// DotUnitary returns the dot product of two vectors as a scalar Matrix.
	DotUnitary(other Matrix) Matrix
	// ClipInPlace clips in place each value of the matrix.
	ClipInPlace(min, max float64) Matrix
	// Maximum returns a new matrix containing the element-wise maxima.
	Maximum(other Matrix) Matrix
	// Minimum returns a new matrix containing the element-wise minima.
	Minimum(other Matrix) Matrix
	// Abs returns a new matrix applying the absolute value function to all elements.
	Abs() Matrix
	// Pow returns a new matrix, applying the power function with given exponent
	// to all elements of the matrix.
	Pow(power float64) Matrix
	// Sqrt returns a new matrix applying the square root function to all elements.
	Sqrt() Matrix
	// Sum returns the sum of all values of the matrix as a scalar Matrix.
	Sum() Matrix
	// Max returns the maximum value of the matrix as a scalar Matrix.
	Max() Matrix
	// Min returns the minimum value of the matrix as a scalar Matrix.
	Min() Matrix
	// ArgMax returns the index of the vector's element with the maximum value.
	ArgMax() int
	// Softmax applies the softmax function to the vector, returning the
	// result as a new column vector.
	Softmax() Matrix
	// CumSum computes the cumulative sum of the vector's elements, returning
	// the result as a new column vector.
	CumSum() Matrix
	// Range creates a new vector initialized with data extracted from the
	// matrix raw data, from start (inclusive) to end (exclusive).
	Range(start, end int) Matrix
	// SplitV splits the vector in N chunks of given sizes,
	// so that N[i] has size sizes[i].
	SplitV(sizes ...int) []Matrix
	// Augment places the identity matrix at the end of the original matrix.
	Augment() Matrix
	// SwapInPlace swaps two rows of the matrix in place.
	SwapInPlace(r1, r2 int) Matrix
	// PadRows returns a copy of the matrix with n additional tail rows.
	// The additional elements are set to zero.
	PadRows(n int) Matrix
	// PadColumns returns a copy of the matrix with n additional tail columns.
	// The additional elements are set to zero.
	PadColumns(n int) Matrix
	// AppendRows returns a copy of the matrix with len(vs) additional tail rows,
	// being each new row filled with the values of each given vector.
	//
	// It accepts row or column vectors indifferently, virtually treating all of
	// them as row vectors.
	AppendRows(vs ...Matrix) Matrix
	// Norm returns the vector's norm. Use pow = 2.0 to compute the Euclidean norm.
	// The result is a scalar Matrix.
	Norm(pow float64) Matrix
	// Pivoting returns the partial pivots of a square matrix to reorder rows.
	// Considerate square sub-matrix from element (offset, offset).
	Pivoting(row int) (Matrix, bool, [2]int)
	// Normalize2 normalizes an array with the Euclidean norm.
	Normalize2() Matrix
	// LU performs lower–upper (LU) decomposition of a square matrix D such as
	// PLU = D, L is lower diagonal and U is upper diagonal, p are pivots.
	LU() (l, u, p Matrix)
	// Inverse returns the inverse of the Matrix.
	Inverse() Matrix
	// Apply creates a new matrix executing the unary function fn.
	Apply(fn func(r, c int, v float64) float64) Matrix
	// ApplyInPlace executes the unary function fn over the matrix a,
	// and stores the result in the receiver, returning the receiver itself.
	ApplyInPlace(fn func(r, c int, v float64) float64, a Matrix) Matrix
	// ApplyWithAlpha creates a new matrix executing the unary function fn,
	// taking additional parameters alpha.
	ApplyWithAlpha(fn func(r, c int, v float64, alpha ...float64) float64, alpha ...float64) Matrix
	// ApplyWithAlphaInPlace executes the unary function fn over the matrix a,
	// taking additional parameters alpha, and stores the result in the
	// receiver, returning the receiver itself.
	ApplyWithAlphaInPlace(fn func(r, c int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64) Matrix
	// DoNonZero calls a function for each non-zero element of the matrix.
	// The parameters of the function are the element's indices and value.
	DoNonZero(fn func(r, c int, v float64))
	// DoVecNonZero calls a function for each non-zero element of the vector.
	// The parameters of the function are the element's index and value.
	DoVecNonZero(fn func(i int, v float64))
	// Clone returns a new matrix, copying all its values from the receiver.
	Clone() Matrix
	// Copy copies the data from the other matrix to the receiver.
	Copy(other Matrix)
	// String returns a string representation of the matrix.
	String() string

	// NewMatrix creates a new matrix, of the same type of the receiver, of
	// size rows×cols, initialized with a copy of raw data.
	//
	// Rows and columns MUST not be negative, and the length of data MUST be
	// equal to rows*cols, otherwise the method panics.
	NewMatrix(rows, cols int, data FloatSliceInterface) Matrix
	// NewVec creates a new column vector (len(data)×1), of the same type of
	// the receiver, initialized with a copy of raw data.
	NewVec(data FloatSliceInterface) Matrix
	// NewScalar creates a new 1×1 matrix, of the same type of the receiver,
	// containing the given value.
	NewScalar(v FloatInterface) Matrix
	// NewEmptyVec creates a new vector, of the same type of the receiver,
	// with dimensions size×1, initialized with zeros.
	NewEmptyVec(size int) Matrix
	// NewEmptyMatrix creates a new rows×cols matrix, of the same type of the
	// receiver, initialized with zeros.
	NewEmptyMatrix(rows, cols int) Matrix
	// NewInitMatrix creates a new rows×cols dense matrix, of the same type
	// of the receiver, initialized with a constant value.
	NewInitMatrix(rows, cols int, v FloatInterface) Matrix
	// NewInitFuncMatrix creates a new rows×cols dense matrix, of the same type
	// of the receiver, initialized with the values returned from the
	// callback function.
	NewInitFuncMatrix(rows, cols int, fn func(r, c int) FloatInterface) Matrix
	// NewInitVec creates a new column vector (size×1), of the same type of
	// the receiver, initialized with a constant value.
	NewInitVec(size int, v FloatInterface) Matrix
	// NewIdentityMatrix creates a new square identity matrix (size×size), of
	// the same type of the receiver, that is, with ones on the diagonal
	// and zeros elsewhere.
	NewIdentityMatrix(size int) Matrix
	// NewOneHotVec creates a new one-hot column vector (size×1), of the same
	// type of the receiver.
	NewOneHotVec(size int, oneAt int) Matrix
	// NewConcatV creates a new column vector, of the same type of the receiver,
	// concatenating two or more vectors "vertically"
	// It accepts row or column vectors indifferently, virtually
	// treating all of them as column vectors.
	NewConcatV(vs ...Matrix) Matrix
	// NewStack creates a new matrix, of the same type of the receiver, stacking
	// two or more vectors of the same size on top of each other; the result is
	// a new matrix where each row contains the data of each input vector.
	// It accepts row or column vectors indifferently, virtually treating all of
	// them as row vectors.
	NewStack(vs ...Matrix) Matrix
}

// Data returns the underlying data of the matrix, as a raw one-dimensional
// slice of values in row-major order.
func Data[T DType](m Matrix) []T {
	return DTFloatSlice[T](m.Data())
}

// SetData sets the content of the matrix, copying the given raw
// data representation as one-dimensional slice.
func SetData[T DType](m Matrix, data []T) {
	m.SetData(FloatSlice(data))
}

// IsVector returns whether the matrix is either a row or column vector
// (dimensions N×1 or 1×N).
func IsVector(m Matrix) bool {
	return m.Rows() == 1 || m.Columns() == 1
}

// IsScalar returns whether the matrix contains exactly one scalar value
// (dimensions 1×1).
func IsScalar(m Matrix) bool {
	return m.Size() == 1
}

// SameDims reports whether the two matrices have the same dimensions.
func SameDims(a, b Matrix) bool {
	return a.Rows() == b.Rows() && a.Columns() == b.Columns()
}

// VectorsOfSameSize reports whether both matrices are vectors (indifferently
// row or column vectors) and have the same size.
func VectorsOfSameSize(a, b Matrix) bool {
	return a.Size() == b.Size() && IsVector(a) && IsVector(b)
}

// ConcatV concatenates two or more vectors "vertically", creating a new Dense
// column vector. It accepts row or column vectors indifferently, virtually
// treating all of them as column vectors.
func ConcatV[T DType](vs ...Matrix) *Dense[T] {
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
func Stack[T DType](vs ...Matrix) *Dense[T] {
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
func Equal(a, b Matrix) bool {
	return a.Rows() == b.Rows() &&
		a.Columns() == b.Columns() &&
		a.Data().Equals(b.Data())
}

// InDelta reports whether matrices a and b have the same shape and
// all elements at the same positions are within delta.
func InDelta(a, b Matrix, delta float64) bool {
	return a.Rows() == b.Rows() &&
		a.Columns() == b.Columns() &&
		a.Data().InDelta(b.Data(), delta)
}
