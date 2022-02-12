// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"encoding/gob"
	"fmt"
	"github.com/nlpodyssey/spago/mat/internal/f32"
	"github.com/nlpodyssey/spago/mat/internal/f32/asm32"
	"github.com/nlpodyssey/spago/mat/internal/f64"
	"github.com/nlpodyssey/spago/mat/internal/f64/asm64"
)

// A Dense matrix implementation.
type Dense[T DType] struct {
	rows  int
	cols  int
	flags denseFlag
	data  []T
}

func init() {
	gob.Register(&Dense[float32]{})
	gob.Register(&Dense[float64]{})
}

// NewDense returns a new matrix of size rows×cols, initialized with a
// copy of raw data.
//
// Rows and columns MUST not be negative, and the length of data MUST be
// equal to rows*cols, otherwise the method panics.
func NewDense[T DType](rows, cols int, data []T) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	if len(data) != rows*cols {
		panic(fmt.Sprintf("mat: wrong matrix dimensions. Elements size must be: %d", rows*cols))
	}
	d := densePool[T]().Get(rows, cols)
	copy(d.data, data)
	return d
}

// NewVecDense returns a new column vector (len(data)×1) initialized with
// a copy of raw data.
func NewVecDense[T DType](data []T) *Dense[T] {
	d := densePool[T]().Get(len(data), 1)
	copy(d.data, data)
	return d
}

// NewScalar returns a new 1×1 matrix containing the given value.
func NewScalar[T DType](v T) *Dense[T] {
	d := densePool[T]().Get(1, 1)
	d.data[0] = v
	return d
}

// NewEmptyVecDense returns a new vector with dimensions size×1, initialized
// with zeros.
func NewEmptyVecDense[T DType](size int) *Dense[T] {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	return densePool[T]().GetEmpty(size, 1)
}

// NewEmptyDense returns a new rows×cols matrix, initialized with zeros.
func NewEmptyDense[T DType](rows, cols int) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	return densePool[T]().GetEmpty(rows, cols)
}

// NewOneHotVecDense returns a new one-hot column vector (size×1).
func NewOneHotVecDense[T DType](size int, oneAt int) *Dense[T] {
	if size <= 0 {
		panic("mat: the vector size must be a positive number")
	}
	if oneAt < 0 || oneAt >= size {
		panic(fmt.Sprintf("mat: impossible to set the one at index %d. The size is: %d", oneAt, size))
	}
	vec := densePool[T]().GetEmpty(size, 1)
	vec.data[oneAt] = 1
	return vec
}

// NewInitDense returns a new rows×cols dense matrix initialized with a
// constant value.
func NewInitDense[T DType](rows, cols int, v T) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	out := densePool[T]().Get(rows, cols)
	data := out.data // avoid bounds check in loop
	for i := range data {
		data[i] = v
	}
	return out
}

// NewInitFuncDense returns a new rows×cols dense matrix initialized with the
// values returned from the callback function.
func NewInitFuncDense[T DType](rows, cols int, fn func(r, c int) T) *Dense[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	out := densePool[T]().Get(rows, cols)

	outData := out.data

	r := 0
	c := 0
	for i := range outData {
		outData[i] = fn(r, c)
		c++
		if c == cols {
			r++
			c = 0
		}
	}

	return out
}

// NewInitVecDense returns a new column vector (size×1) initialized with a
// constant value.
func NewInitVecDense[T DType](size int, v T) *Dense[T] {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	out := densePool[T]().Get(size, 1)
	data := out.data // avoid bounds check in loop
	for i := range data {
		data[i] = v
	}
	return out
}

// NewIdentityDense returns a square identity matrix (size×size), that is,
// with ones on the diagonal and zeros elsewhere.
func NewIdentityDense[T DType](size int) *Dense[T] {
	if size < 0 {
		panic("mat: a negative size is not allowed")
	}
	out := densePool[T]().GetEmpty(size, size)
	data := out.data
	ln := len(data)
	for i := 0; i < ln; i += size + 1 {
		data[i] = 1
	}
	return out
}

// Rows returns the number of rows of the matrix.
func (d *Dense[_]) Rows() int {
	return d.rows
}

// Columns returns the number of columns of the matrix.
func (d *Dense[_]) Columns() int {
	return d.cols
}

// Dims returns the number of rows and columns of the matrix.
func (d *Dense[_]) Dims() (r, c int) {
	return d.rows, d.cols
}

// The Size of the matrix (rows*columns).
func (d *Dense[_]) Size() int {
	return len(d.data)
}

// Data returns the underlying data of the matrix, as a raw one-dimensional
// slice of values in row-major order.
//
// The data slice IS NOT a copy: any changes applied to the returned slice are
// reflected in the Dense matrix too.
func (d *Dense[T]) Data() []T {
	return d.data
}

// SetData sets the content of the matrix, copying the given raw
// data representation as one-dimensional slice.
func (d *Dense[T]) SetData(data []T) {
	if len(data) != len(d.data) {
		panic(fmt.Sprintf("mat: incompatible data size. Expected: %d Found: %d", len(d.data), len(data)))
	}
	copy(d.data, data)
}

// ZerosLike returns a new matrix with the same dimensions of the
// receiver, initialized with zeroes.
func (d *Dense[T]) ZerosLike() Matrix[T] {
	return NewEmptyDense[T](d.rows, d.cols)
}

// OnesLike returns a new matrix with the same dimensions of the
// receiver, initialized with ones.
func (d *Dense[T]) OnesLike() Matrix[T] {
	out := densePool[T]().Get(d.rows, d.cols)
	data := out.data // avoid bounds check in loop
	for i := range data {
		data[i] = 1.0
	}
	return out
}

// Scalar returns the scalar value.
// It panics if the matrix does not contain exactly one element.
func (d *Dense[T]) Scalar() T {
	if !IsScalar(Matrix[T](d)) {
		panic("mat: expected scalar but the matrix contains more elements")
	}
	return d.data[0]
}

// Zeros sets all the values of the matrix to zero.
func (d *Dense[T]) Zeros() {
	data := d.data // avoid bounds check in loop
	for i := range data {
		data[i] = T(0)
	}
}

// Set sets the value v at row r and column c.
// It panics if the given indices are out of range.
func (d *Dense[T]) Set(r int, c int, v T) {
	if r < 0 || r >= d.rows {
		panic("mat: 'r' argument out of range")
	}
	if c < 0 || c >= d.cols {
		panic("mat: 'c' argument out of range")
	}
	d.data[r*d.cols+c] = v
}

// At returns the value at row r and column c.
// It panics if the given indices are out of range.
func (d *Dense[T]) At(r int, c int) T {
	if r < 0 || r >= d.rows {
		panic("mat: 'r' argument out of range")
	}
	if c < 0 || c >= d.cols {
		panic("mat: 'c' argument out of range")
	}
	return d.data[r*d.cols+c]
}

// SetVec sets the value v at position i of a vector.
// It panics if the receiver is not a vector or the position is out of range.
func (d *Dense[T]) SetVec(i int, v T) {
	if !(IsVector(Matrix[T](d))) {
		panic("mat: expected vector")
	}
	if i < 0 || i >= len(d.data) {
		panic("mat: 'i' argument out of range")
	}
	d.data[i] = v
}

// AtVec returns the value at position i of a vector.
// It panics if the receiver is not a vector or the position is out of range.
func (d *Dense[T]) AtVec(i int) T {
	if !IsVector(Matrix[T](d)) {
		panic("mat: expected vector")
	}
	if i < 0 || i >= len(d.data) {
		panic("mat: 'i' argument out of range")
	}
	return d.data[i]
}

// ExtractRow returns a copy of the i-th row of the matrix.
func (d *Dense[T]) ExtractRow(i int) Matrix[T] {
	if i < 0 || i >= d.rows {
		panic("mat: index out of range")
	}
	start := i * d.cols
	out := NewVecDense[T](d.data[start : start+d.cols])
	return out
}

// ExtractColumn returns a copy of the i-th column of the matrix.
func (d *Dense[T]) ExtractColumn(i int) Matrix[T] {
	if i < 0 || i >= d.cols {
		panic("mat: index out of range")
	}
	out := densePool[T]().Get(d.rows, 1)
	dData := d.data
	outData := out.data
	for k := range outData {
		outData[k] = dData[k*d.cols+i]
	}
	return out
}

// View returns a new Matrix sharing the same underlying data.
func (d *Dense[T]) View(rows, cols int) Matrix[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	if rows*cols != len(d.data) {
		panic(fmt.Sprintf("mat: wrong matrix dimensions. Size (rows*cols) must be: %d", len(d.data)))
	}
	return &Dense[T]{
		rows:  rows,
		cols:  cols,
		flags: denseIsView,
		data:  d.data,
	}
}

// Reshape returns a copy of the matrix.
// It panics if the dimensions are incompatible.
func (d *Dense[T]) Reshape(rows, cols int) Matrix[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	if rows*cols != len(d.data) {
		panic(fmt.Sprintf("mat: wrong matrix dimensions. Size (rows*cols) must be: %d", len(d.data)))
	}
	return NewDense(rows, cols, d.data)
}

// ReshapeInPlace changes the dimensions of the matrix in place and returns the
// matrix itself.
// It panics if the dimensions are incompatible.
func (d *Dense[T]) ReshapeInPlace(rows, cols int) Matrix[T] {
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	if rows*cols != len(d.data) {
		panic(fmt.Sprintf("mat: wrong matrix dimensions. Size (rows*cols) must be: %d", len(d.data)))
	}
	d.rows = rows
	d.cols = cols
	return d
}

// ResizeVector returns a resized copy of the vector.
//
// If the new size is smaller than the input vector, the remaining tail
// elements are removed. If it's bigger, the additional tail elements
// are set to zero.
func (d *Dense[T]) ResizeVector(newSize int) Matrix[T] {
	if newSize < 0 {
		panic("mat: a negative size is not allowed")
	}
	dSize := len(d.data)
	if newSize <= dSize {
		return NewVecDense(d.data[:newSize])
	}

	y := NewEmptyVecDense[T](newSize)
	copy(y.data[:dSize], d.data)
	return y
}

// T returns the transpose of the matrix.
func (d *Dense[T]) T() Matrix[T] {
	dRows := d.rows
	dCols := d.cols

	m := densePool[T]().Get(dCols, dRows)
	size := len(m.data)
	index := 0
	for _, value := range d.data {
		m.data[index] = value
		index += dRows
		if index >= size {
			index -= size - 1
		}
	}
	return m
}

// Prod performs the element-wise product between the receiver and the other matrix.
func (d *Dense[T]) Prod(other Matrix[T]) Matrix[T] {
	if !(SameDims(Matrix[T](d), other) || VectorsOfSameSize(Matrix[T](d), other)) {
		panic("mat: matrices have incompatible dimensions")
	}

	out := densePool[T]().Get(d.rows, d.cols)

	// Avoid bounds checks in loop
	dData := d.data
	oData := other.Data()
	outData := out.data
	lastIndex := len(oData) - 1
	if lastIndex < 0 {
		return out
	}
	_ = outData[lastIndex]
	_ = dData[lastIndex]
	for i := lastIndex; i >= 0; i-- {
		outData[i] = dData[i] * oData[i]
	}
	return out
}

// ProdInPlace performs the in-place element-wise product with the other matrix.
func (d *Dense[T]) ProdInPlace(other Matrix[T]) Matrix[T] {
	if !(SameDims(Matrix[T](d), other) || VectorsOfSameSize(Matrix[T](d), other)) {
		panic("mat: matrices have incompatible dimensions")
	}
	dData := d.data
	oData := other.Data()
	for i, val := range oData {
		dData[i] *= val
	}
	return d
}

// DivInPlace performs the in-place element-wise division of the receiver by the other matrix.
func (d *Dense[T]) DivInPlace(other Matrix[T]) Matrix[T] {
	if !(SameDims(Matrix[T](d), other) || VectorsOfSameSize(Matrix[T](d), other)) {
		panic("mat: matrices have incompatible dimensions")
	}
	for i, val := range other.Data() {
		d.data[i] *= 1.0 / val
	}
	return d
}

// ClipInPlace clips in place each value of the matrix.
func (d *Dense[T]) ClipInPlace(min, max T) Matrix[T] {
	data := d.data
	for i, v := range data {
		switch {
		case v < min:
			data[i] = min
		case v > max:
			data[i] = max
		default:
			continue
		}
	}
	return d
}

// Max returns the maximum value of the matrix.
func (d *Dense[T]) Max() T {
	if len(d.data) == 0 {
		panic("mat: cannot find the maximum value in an empty matrix")
	}
	max := d.data[0]
	for _, v := range d.data[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// Min returns the minimum value of the matrix.
func (d *Dense[T]) Min() T {
	if len(d.data) == 0 {
		panic("mat: cannot find the minimum value in an empty matrix")
	}
	min := d.data[0]
	for _, v := range d.data[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

// Maximum returns a new matrix containing the element-wise maxima.
func (d *Dense[T]) Maximum(other Matrix[T]) Matrix[T] {
	if !SameDims(Matrix[T](d), other) {
		panic("mat: matrices have incompatible dimensions")
	}
	out := densePool[T]().Get(d.rows, d.cols)
	dData := d.data
	otherData := other.Data()
	outData := out.data
	for i := range outData {
		dV := dData[i]
		otherV := otherData[i]
		if dV > otherV {
			outData[i] = dV
			continue
		}
		outData[i] = otherV
	}
	return out
}

// Minimum returns a new matrix containing the element-wise minima.
func (d *Dense[T]) Minimum(other Matrix[T]) Matrix[T] {
	if !SameDims(Matrix[T](d), other) {
		panic("mat: matrices have incompatible dimensions")
	}
	out := densePool[T]().Get(d.rows, d.cols)
	dData := d.data
	otherData := other.Data()
	outData := out.data
	for i := range outData {
		dV := dData[i]
		otherV := otherData[i]
		if dV < otherV {
			outData[i] = dV
			continue
		}
		outData[i] = otherV
	}
	return out
}

// Range creates a new vector initialized with data extracted from the
// matrix raw data, from start (inclusive) to end (exclusive).
func (d *Dense[T]) Range(start, end int) Matrix[T] {
	return NewVecDense(d.data[start:end])
}

// SplitV extract N vectors from the Matrix.
// N[i] has size sizes[i].
func (d *Dense[T]) SplitV(sizes ...int) []Matrix[T] {
	if len(sizes) == 0 {
		return nil
	}
	out := make([]Matrix[T], len(sizes))
	offset := 0
	for i, size := range sizes {
		if size < 0 {
			panic("mat: a negative size is not allowed")
		}
		startIndex := offset
		offset = startIndex + size
		out[i] = d.Range(startIndex, offset)
	}
	return out
}

// Apply creates a new matrix executing the unary function fn.
func (d *Dense[T]) Apply(fn func(r, c int, v T) T) Matrix[T] {
	out := densePool[T]().Get(d.rows, d.cols)
	if len(d.data) == 0 {
		return out
	}

	dData := d.data
	outData := out.data
	_ = outData[len(dData)-1]

	r := 0
	c := 0
	for i, v := range dData {
		outData[i] = fn(r, c, v)
		c++
		if c == d.cols {
			r++
			c = 0
		}
	}

	return out
}

// ApplyInPlace executes the unary function fn.
func (d *Dense[T]) ApplyInPlace(fn func(r, c int, v T) T, a Matrix[T]) {
	if !SameDims(Matrix[T](d), a) {
		panic("mat: incompatible matrix dimensions")
	}
	aData := a.Data()
	lastIndex := len(aData) - 1
	if lastIndex < 0 {
		return
	}
	r := 0
	c := 0
	dData := d.data
	_ = dData[lastIndex]
	for i, val := range aData {
		dData[i] = fn(r, c, val)
		c++
		if c == d.cols {
			r++
			c = 0
		}
	}
}

// ApplyWithAlpha creates a new matrix executing the unary function fn,
// taking additional parameters alpha.
func (d *Dense[T]) ApplyWithAlpha(fn func(r, c int, v T, alpha ...T) T, alpha ...T) Matrix[T] {
	out := densePool[T]().Get(d.rows, d.cols)
	if len(d.data) == 0 {
		return out
	}

	dData := d.data
	outData := out.data
	_ = outData[len(dData)-1]

	r := 0
	c := 0
	for i, v := range dData {
		outData[i] = fn(r, c, v, alpha...)
		c++
		if c == d.cols {
			r++
			c = 0
		}
	}

	return out
}

// ApplyWithAlphaInPlace executes the unary function fn, taking additional parameters alpha.
func (d *Dense[T]) ApplyWithAlphaInPlace(fn func(r, c int, v T, alpha ...T) T, a Matrix[T], alpha ...T) {
	if !SameDims(Matrix[T](d), a) {
		panic("mat: incompatible matrix dimensions")
	}
	// TODO: rewrite for better performance
	for r := 0; r < d.rows; r++ {
		for c := 0; c < d.cols; c++ {
			d.data[r*d.cols+c] = fn(r, c, a.At(r, c), alpha...)
		}
	}
}

// DoNonZero calls a function for each non-zero element of the matrix.
// The parameters of the function are the element's indices and value.
func (d *Dense[T]) DoNonZero(fn func(r, c int, v T)) {
	for r, di := 0, 0; r < d.rows; r++ {
		for c := 0; c < d.cols; c, di = c+1, di+1 {
			v := d.data[di]
			if v == 0 {
				continue
			}
			fn(r, c, v)
		}
	}
}

// DoVecNonZero calls a function for each non-zero element of the vector.
// The parameters of the function are the element's index and value.
func (d *Dense[T]) DoVecNonZero(fn func(i int, v T)) {
	if !IsVector(Matrix[T](d)) {
		panic("mat: expected vector")
	}
	for i, v := range d.data {
		if v == 0 {
			continue
		}
		fn(i, v)
	}
}

// Augment places the identity matrix at the end of the original matrix.
func (d *Dense[T]) Augment() Matrix[T] {
	if d.cols != d.rows {
		panic("mat: matrix must be square")
	}
	// TODO: rewrite for better performance
	out := NewEmptyDense[T](d.rows, d.rows+d.cols)
	for i := 0; i < d.rows; i++ {
		for j := 0; j < d.cols; j++ {
			out.Set(i, j, d.At(i, j))
		}
		out.Set(i, i+d.rows, 1.0)
	}
	return out
}

// SwapInPlace swaps two rows of the matrix in place.
func (d *Dense[T]) SwapInPlace(r1, r2 int) Matrix[T] {
	if r1 < 0 || r1 >= d.rows {
		panic("mat: 'r1' argument out of range")
	}
	if r2 < 0 || r2 >= d.rows {
		panic("mat: 'r2' argument out of range")
	}
	// TODO: rewrite for better performance
	for j := 0; j < d.cols; j++ {
		a, b := r1*d.cols+j, r2*d.cols+j
		d.data[a], d.data[b] = d.data[b], d.data[a]
	}
	return d
}

// PadRows returns a copy of the matrix with n additional tail rows.
// The additional elements are set to zero.
func (d *Dense[T]) PadRows(n int) Matrix[T] {
	if n < 0 {
		panic("mat: negative 'n' argument is not allowed")
	}
	cols := d.cols
	dRows := d.rows
	yRows := dRows + n
	y := NewEmptyDense[T](yRows, cols)

	if cols == 0 || dRows == 0 {
		return y
	}

	dData := d.data
	yData := y.data
	copy(yData[:len(dData)], dData)

	return y
}

// PadColumns returns a copy of the matrix with n additional tail columns.
// The additional elements are set to zero.
func (d *Dense[T]) PadColumns(n int) Matrix[T] {
	if n < 0 {
		panic("mat: negative 'n' argument is not allowed")
	}
	rows := d.rows
	dCols := d.cols
	yCols := dCols + n
	y := NewEmptyDense[T](rows, yCols)

	if rows == 0 || dCols == 0 {
		return y
	}

	dData := d.data
	yData := y.data
	for r, xi, yi := 0, 0, 0; r < rows; r, xi, yi = r+1, xi+dCols, yi+yCols {
		copy(yData[yi:yi+dCols], dData[xi:xi+dCols])
	}

	return y
}

// Abs returns a new matrix applying the absolute value function to all elements.
func (d *Dense[T]) Abs() Matrix[T] {
	out := densePool[T]().Get(d.rows, d.cols)
	outData := out.data
	for i, val := range d.data {
		outData[i] = Abs(val)
	}
	return out
}

// Pow returns a new matrix, applying the power function with given exponent
// to all elements of the matrix.
func (d *Dense[T]) Pow(power T) Matrix[T] {
	out := densePool[T]().Get(d.rows, d.cols)
	outData := out.data
	for i, val := range d.data {
		outData[i] = Pow(val, power)
	}
	return out
}

// Sqrt returns a new matrix applying the square root function to all elements.
func (d *Dense[T]) Sqrt() Matrix[T] {
	out := densePool[T]().Get(d.rows, d.cols)
	inData := d.data
	lastIndex := len(inData) - 1
	if lastIndex < 0 {
		return out
	}
	outData := out.data
	_ = outData[lastIndex]
	for i, val := range inData {
		outData[i] = Sqrt(val)
	}
	return out
}

// Norm returns the vector's norm. Use pow = 2.0 to compute the Euclidean norm.
func (d *Dense[T]) Norm(pow T) T {
	var s T
	for _, x := range d.data {
		s += Pow(x, pow)
	}
	return Pow(s, 1/pow)
}

// Pivoting returns the partial pivots of a square matrix to reorder rows.
// Considerate square sub-matrix from element (offset, offset).
func (d *Dense[T]) Pivoting(row int) (Matrix[T], bool, []int) {
	if d.rows != d.cols {
		panic("mat: matrix must be square")
	}
	pv := make([]int, d.cols)
	positions := make([]int, 2)
	for i := range pv {
		pv[i] = i
	}
	j := row
	max := Abs(d.data[row*d.cols+j])
	for i := row; i < d.cols; i++ {
		if d.data[i*d.cols+j] > max {
			max = Abs(d.data[i*d.cols+j])
			row = i
		}
	}
	swap := false
	if j != row {
		pv[row], pv[j] = pv[j], pv[row]
		swap = true
		positions[0] = row
		positions[1] = j
	}

	p := NewEmptyDense[T](d.cols, d.cols)
	for r, c := range pv {
		p.data[r*d.cols+c] = 1
	}
	return p, swap, positions
}

// AddScalar performs the addition between the matrix and the given value.
func (d *Dense[T]) AddScalar(n T) Matrix[T] {
	out := NewDense(d.rows, d.cols, d.data)
	switch nt := any(n).(type) {
	case float32:
		f32.AddConst(nt, any(out.data).([]float32))
	case float64:
		asm64.AddConst(nt, any(out.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// AddScalarInPlace adds the scalar to all values of the matrix.
func (d *Dense[T]) AddScalarInPlace(n T) Matrix[T] {
	switch nt := any(n).(type) {
	case float32:
		f32.AddConst(nt, any(d.data).([]float32))
	case float64:
		asm64.AddConst(nt, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// SubScalar performs a subtraction between the matrix and the given value.
func (d *Dense[T]) SubScalar(n T) Matrix[T] {
	out := NewDense(d.rows, d.cols, d.data)
	switch nt := any(n).(type) {
	case float32:
		f32.AddConst(-nt, any(out.data).([]float32))
	case float64:
		asm64.AddConst(-nt, any(out.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// SubScalarInPlace subtracts the scalar from the receiver's values.
func (d *Dense[T]) SubScalarInPlace(n T) Matrix[T] {
	switch nt := any(n).(type) {
	case float32:
		f32.AddConst(-nt, any(d.data).([]float32))
	case float64:
		asm64.AddConst(-nt, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// ProdScalar returns the multiplication between the matrix and the given value.
func (d *Dense[T]) ProdScalar(n T) Matrix[T] {
	out := NewEmptyDense[T](d.rows, d.cols)
	switch nt := any(n).(type) {
	case float32:
		asm32.ScalUnitaryTo(any(out.data).([]float32), nt, any(d.data).([]float32))
	case float64:
		asm64.ScalUnitaryTo(any(out.data).([]float64), nt, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// ProdScalarInPlace performs the in-place multiplication between the
// matrix and the given value.
func (d *Dense[T]) ProdScalarInPlace(n T) Matrix[T] {
	switch nt := any(n).(type) {
	case float32:
		asm32.ScalUnitary(nt, any(d.data).([]float32))
	case float64:
		asm64.ScalUnitary(nt, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// Add returns the addition between the receiver and another matrix.
func (d *Dense[T]) Add(other Matrix[T]) Matrix[T] {
	if !(SameDims(Matrix[T](d), other) || VectorsOfSameSize(Matrix[T](d), other)) {
		panic("mat: matrices have incompatible dimensions")
	}
	out := NewEmptyDense[T](d.rows, d.cols)
	switch any(T(0)).(type) {
	case float32:
		asm32.AxpyUnitaryTo(any(out.data).([]float32), 1, any(other.Data()).([]float32), any(d.data).([]float32))
	case float64:
		asm64.AxpyUnitaryTo(any(out.data).([]float64), 1, any(other.Data()).([]float64), any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// AddInPlace performs the in-place addition with the other matrix.
func (d *Dense[T]) AddInPlace(other Matrix[T]) Matrix[T] {
	if !(SameDims(Matrix[T](d), other) || VectorsOfSameSize(Matrix[T](d), other)) {
		panic("mat: matrices have incompatible dimensions")
	}
	switch any(T(0)).(type) {
	case float32:
		asm32.AxpyUnitary(1, any(other.Data()).([]float32), any(d.data).([]float32))
	case float64:
		asm64.AxpyUnitary(1, any(other.Data()).([]float64), any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// Sub returns the subtraction of the other matrix from the receiver.
func (d *Dense[T]) Sub(other Matrix[T]) Matrix[T] {
	if !(SameDims(Matrix[T](d), other) || VectorsOfSameSize(Matrix[T](d), other)) {
		panic("mat: matrices have incompatible dimensions")
	}
	out := NewEmptyDense[T](d.rows, d.cols)
	switch any(T(0)).(type) {
	case float32:
		asm32.AxpyUnitaryTo(any(out.data).([]float32), -1, any(other.Data()).([]float32), any(d.data).([]float32))
	case float64:
		asm64.AxpyUnitaryTo(any(out.data).([]float64), -1, any(other.Data()).([]float64), any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// SubInPlace performs the in-place subtraction with the other matrix.
func (d *Dense[T]) SubInPlace(other Matrix[T]) Matrix[T] {
	if !(SameDims(Matrix[T](d), other) || VectorsOfSameSize(Matrix[T](d), other)) {
		panic("mat: matrices have incompatible dimensions")
	}
	switch any(T(0)).(type) {
	case float32:
		asm32.AxpyUnitary(-1, any(other.Data()).([]float32), any(d.data).([]float32))
	case float64:
		asm64.AxpyUnitary(-1, any(other.Data()).([]float64), any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// ProdMatrixScalarInPlace multiplies the given matrix with the value,
// storing the result in the receiver.
func (d *Dense[T]) ProdMatrixScalarInPlace(m Matrix[T], n T) Matrix[T] {
	switch nt := any(n).(type) {
	case float32:
		asm32.ScalUnitaryTo(any(d.data).([]float32), nt, any(m.Data()).([]float32))
	case float64:
		asm64.ScalUnitaryTo(any(d.data).([]float64), nt, any(m.Data()).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// Div returns the result of the element-wise division of the receiver by the other matrix.
func (d *Dense[T]) Div(other Matrix[T]) Matrix[T] {
	if !(SameDims(Matrix[T](d), other) || VectorsOfSameSize(Matrix[T](d), other)) {
		panic("mat: matrices have incompatible dimensions")
	}
	out := NewEmptyDense[T](d.rows, d.cols)
	switch any(T(0)).(type) {
	case float32:
		f32.DivTo(any(out.data).([]float32), any(d.data).([]float32), any(other.Data()).([]float32))
	case float64:
		asm64.DivTo(any(out.data).([]float64), any(d.data).([]float64), any(other.Data()).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// Mul performs the multiplication row by column.
// If A is an i×j Matrix, and B is j×k, then the resulting Matrix
// C = AB will be i×k.
func (d *Dense[T]) Mul(other Matrix[T]) Matrix[T] {
	if d.cols != other.Rows() {
		panic("mat: matrices have incompatible dimensions")
	}
	out := densePool[T]().GetEmpty(d.rows, other.Columns())

	switch any(T(0)).(type) {
	case float32:
		if out.cols != 1 {
			f32.MatrixMul(
				d.rows,                        // aRows
				d.cols,                        // aCols
				other.Columns(),               // bCols
				any(d.data).([]float32),       // a
				any(other.Data()).([]float32), // b
				any(out.data).([]float32),     // c
			)
			return out
		}

		asm32.GemvN(
			uintptr(d.rows),               // m
			uintptr(d.cols),               // n
			1,                             // alpha
			any(d.data).([]float32),       // a
			uintptr(d.cols),               // lda
			any(other.Data()).([]float32), // x
			1,                             // incX
			0,                             // beta
			any(out.data).([]float32),     // y
			1,                             // incY
		)
	case float64:
		if out.cols != 1 {
			f64.MatrixMul(
				d.rows,                        // aRows
				d.cols,                        // aCols
				other.Columns(),               // bCols
				any(d.data).([]float64),       // a
				any(other.Data()).([]float64), // b
				any(out.data).([]float64),     // c
			)
			return out
		}

		asm64.GemvN(
			uintptr(d.rows),               // m
			uintptr(d.cols),               // n
			1,                             // alpha
			any(d.data).([]float64),       // a
			uintptr(d.cols),               // lda
			any(other.Data()).([]float64), // x
			1,                             // incX
			0,                             // beta
			any(out.data).([]float64),     // y
			1,                             // incY
		)
		return out
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// MulT performs the matrix multiplication row by column.
// ATB = C, where AT is the transpose of B
// if A is an r x c Matrix, and B is j x k, r = j the resulting
// Matrix C will be c x k.
func (d *Dense[T]) MulT(other Matrix[T]) Matrix[T] {
	if d.rows != other.Rows() {
		panic("mat: matrices have incompatible dimensions")
	}
	if other.Columns() != 1 {
		panic("mat: the other matrix must have exactly 1 column")
	}
	out := densePool[T]().GetEmpty(d.cols, other.Columns())

	switch any(T(0)).(type) {
	case float32:
		asm32.GemvT(
			uintptr(d.rows),               // m
			uintptr(d.cols),               // n
			1,                             // alpha
			any(d.data).([]float32),       // a
			uintptr(d.cols),               // lda
			any(other.Data()).([]float32), // x
			1,                             // incX
			0,                             // beta
			any(out.data).([]float32),     // y
			1,                             // incY
		)
	case float64:
		asm64.GemvT(
			uintptr(d.rows),               // m
			uintptr(d.cols),               // n
			1,                             // alpha
			any(d.data).([]float64),       // a
			uintptr(d.cols),               // lda
			any(other.Data()).([]float64), // x
			1,                             // incX
			0,                             // beta
			any(out.data).([]float64),     // y
			1,                             // incY
		)
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// DotUnitary returns the dot product of two vectors.
func (d *Dense[T]) DotUnitary(other Matrix[T]) T {
	if d.Size() != other.Size() {
		panic("mat: matrices have incompatible sizes")
	}
	switch any(T(0)).(type) {
	case float32:
		return T(asm32.DotUnitary(any(d.data).([]float32), any(other.Data()).([]float32)))
	case float64:
		return T(asm64.DotUnitary(any(d.data).([]float64), any(other.Data()).([]float64)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Sum returns the sum of all values of the matrix.
func (d *Dense[T]) Sum() T {
	switch any(T(0)).(type) {
	case float32:
		return T(asm32.Sum(any(d.data).([]float32)))
	case float64:
		return T(asm64.Sum(any(d.data).([]float64)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Normalize2 normalizes an array with the Euclidean norm.
func (d *Dense[T]) Normalize2() Matrix[T] {
	norm2 := d.Norm(2)
	if norm2 == 0.0 {
		return d.Clone()
	}
	return d.ProdScalar(1 / norm2)
}

// LU performs lower–upper (LU) decomposition of a square matrix D such as
// PLU = D, L is lower diagonal and U is upper diagonal, p are pivots.
func (d *Dense[T]) LU() (l, u, p Matrix[T]) {
	if d.rows != d.cols {
		panic("mat: matrix must be square")
	}
	u = NewDense(d.rows, d.cols, d.data)
	p = NewIdentityDense[T](d.cols)
	l = NewEmptyDense[T](d.cols, d.cols)
	lData := l.Data()
	for i := 0; i < d.cols; i++ {
		_, swap, positions := u.Pivoting(i)
		if swap {
			u.SwapInPlace(positions[0], positions[1])
			p.SwapInPlace(positions[0], positions[1])
			l.SwapInPlace(positions[0], positions[1])
		}
		lt := NewIdentityDense[T](d.cols)
		ltData := lt.data
		uData := u.Data()
		for k := i + 1; k < d.cols; k++ {
			ltData[k*d.cols+i] = -uData[k*d.cols+i] / (uData[i*d.cols+i])
			lData[k*d.cols+i] = uData[k*d.cols+i] / (uData[i*d.cols+i])
		}
		u = lt.Mul(u)
	}
	for i := 0; i < d.cols; i++ {
		lData[i*d.cols+i] = 1.0
	}
	return
}

// Inverse returns the inverse of the Matrix.
func (d *Dense[T]) Inverse() Matrix[T] {
	if d.cols != d.rows {
		panic("mat: matrix must be square")
	}
	out := NewEmptyDense[T](d.cols, d.cols)
	outData := out.data
	s := NewEmptyDense[T](d.cols, d.cols)
	sData := s.data
	l, u, p := d.LU()
	lData := l.Data()
	uData := u.Data()
	pData := p.Data()
	for b := 0; b < d.cols; b++ {
		// find solution of Ly = b
		for i := 0; i < l.Rows(); i++ {
			var sum T
			for j := 0; j < i; j++ {
				sum += lData[i*d.cols+j] * sData[j*d.cols+b]
			}
			sData[i*d.cols+b] = pData[i*d.cols+b] - sum
		}
		// find solution of Ux = y
		for i := d.cols - 1; i >= 0; i-- {
			var sum T
			for j := i + 1; j < d.cols; j++ {
				sum += uData[i*d.cols+j] * outData[j*d.cols+b]
			}
			outData[i*d.cols+b] = (1.0 / uData[i*d.cols+i]) * (sData[i*d.cols+b] - sum)
		}
	}
	return out
}

// Clone returns a new matrix, copying all its values from the receiver.
func (d *Dense[T]) Clone() Matrix[T] {
	out := densePool[T]().Get(d.rows, d.cols)
	copy(out.data, d.data)
	return out
}

// Copy copies the data from the other matrix to the receiver.
// It panics if the matrices have different dimensions.
func (d *Dense[T]) Copy(other Matrix[T]) {
	if !SameDims(Matrix[T](d), other) {
		panic("mat: incompatible matrix dimensions")
	}
	copy(d.data, other.Data())
}

// String returns a string representation of the matrix.
func (d *Dense[_]) String() string {
	return fmt.Sprintf("%v", d.data)
}
