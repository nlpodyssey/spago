// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"math"

	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/internal/f32"
	"github.com/nlpodyssey/spago/mat/internal/f32/asm32"
	"github.com/nlpodyssey/spago/mat/internal/f64"
	"github.com/nlpodyssey/spago/mat/internal/f64/asm64"
	"github.com/nlpodyssey/spago/mat/internal/matfuncs"
)

// A Dense matrix implementation.
type Dense[T float.DType] struct {
	rows  int
	cols  int
	flags denseFlag
	data  []T
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
func (d *Dense[T]) Data() float.Slice {
	return float.SliceInterface(d.data)
}

// SetData sets the content of the matrix, copying the given raw
// data representation as one-dimensional slice.
func (d *Dense[T]) SetData(data float.Slice) {
	v := float.SliceValueOf[T](data)
	if len(v) != len(d.data) {
		panic(fmt.Sprintf("mat: incompatible data size, expected %d, actual %d", len(d.data), len(v)))
	}
	copy(d.data, v)
}

// ZerosLike returns a new matrix with the same dimensions of the
// receiver, initialized with zeroes.
func (d *Dense[T]) ZerosLike() Matrix {
	return NewEmptyDense[T](d.rows, d.cols)
}

// OnesLike returns a new matrix with the same dimensions of the
// receiver, initialized with ones.
func (d *Dense[T]) OnesLike() Matrix {
	out := densePool[T]().Get(d.rows, d.cols)
	data := out.data // avoid bounds check in loop
	for i := range data {
		data[i] = 1.0
	}
	return out
}

// Scalar returns the scalar value.
// It panics if the matrix does not contain exactly one element.
func (d *Dense[T]) Scalar() float.Float {
	if !IsScalar(d) {
		panic("mat: expected scalar but the matrix contains more elements")
	}
	return float.Interface(d.data[0])
}

// Zeros sets all the values of the matrix to zero.
func (d *Dense[T]) Zeros() {
	data := d.data // avoid bounds check in loop
	for i := range data {
		data[i] = T(0)
	}
}

// Set sets the scalar value from a 1×1 matrix at row r and column c.
// It panics if the given matrix is not 1×1, or if indices are out of range.
func (d *Dense[T]) Set(r int, c int, m Matrix) {
	d.set(r, c, float.ValueOf[T](m.Scalar()))
}

// At returns the value at row r and column c as a 1×1 matrix.
// It panics if the given indices are out of range.
func (d *Dense[T]) At(r int, c int) Matrix {
	return NewScalar[T](d.at(r, c))
}

// SetScalar sets the value v at row r and column c.
// It panics if the given indices are out of range.
func (d *Dense[T]) SetScalar(r int, c int, v float.Float) {
	d.set(r, c, float.ValueOf[T](v))
}

// ScalarAt returns the value at row r and column c.
// It panics if the given indices are out of range.
func (d *Dense[T]) ScalarAt(r int, c int) float.Float {
	return float.Interface(d.at(r, c))
}

func (d *Dense[T]) set(r int, c int, v T) {
	if r < 0 || r >= d.rows {
		panic("mat: 'r' argument out of range")
	}
	if c < 0 || c >= d.cols {
		panic("mat: 'c' argument out of range")
	}
	d.data[r*d.cols+c] = v
}

func (d *Dense[T]) at(r int, c int) T {
	if r < 0 || r >= d.rows {
		panic("mat: 'r' argument out of range")
	}
	if c < 0 || c >= d.cols {
		panic("mat: 'c' argument out of range")
	}
	return d.data[r*d.cols+c]
}

// SetVec sets the scalar value from a 1×1 matrix at position i of a
// vector. It panics if the receiver is not a vector, or the given matrix is
// not 1×1, or the position is out of range.
func (d *Dense[T]) SetVec(i int, m Matrix) {
	d.setVec(i, float.ValueOf[T](m.Scalar()))
}

// AtVec returns the value at position i of a vector as a 1×1 matrix.
// It panics if the receiver is not a vector or the position is out of range.
func (d *Dense[T]) AtVec(i int) Matrix {
	return NewScalar[T](d.atVec(i))
}

// SetVecScalar sets the value v at position i of a vector.
// It panics if the receiver is not a vector or the position is out of range.
func (d *Dense[T]) SetVecScalar(i int, v float.Float) {
	d.setVec(i, float.ValueOf[T](v))
}

// ScalarAtVec returns the value at position i of a vector.
// It panics if the receiver is not a vector or the position is out of range.
func (d *Dense[T]) ScalarAtVec(i int) float.Float {
	return float.Interface(d.atVec(i))
}

func (d *Dense[T]) setVec(i int, v T) {
	if !(IsVector(d)) {
		panic("mat: expected vector")
	}
	if i < 0 || i >= len(d.data) {
		panic("mat: 'i' argument out of range")
	}
	d.data[i] = v
}

func (d *Dense[T]) atVec(i int) T {
	if !IsVector(d) {
		panic("mat: expected vector")
	}
	if i < 0 || i >= len(d.data) {
		panic("mat: 'i' argument out of range")
	}
	return d.data[i]
}

// ExtractRow returns a copy of the i-th row of the matrix,
// as a row vector (1×cols).
func (d *Dense[T]) ExtractRow(i int) Matrix {
	if i < 0 || i >= d.rows {
		panic("mat: index out of range")
	}
	out := densePool[T]().Get(1, d.cols)
	start := i * d.cols
	copy(out.data, d.data[start:start+d.cols])
	return out
}

// ExtractColumn returns a copy of the i-th column of the matrix,
// as a column vector (rows×1).
func (d *Dense[T]) ExtractColumn(i int) Matrix {
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
func (d *Dense[T]) View(rows, cols int) Matrix {
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

// Slice returns a new matrix obtained by slicing the receiver across the
// given positions. The parameters "fromRow" and "fromCol" are inclusive,
// while "toRow" and "toCol" are exclusive.
func (d *Dense[T]) Slice(fromRow, fromCol, toRow, toCol int) Matrix {
	dRows := d.rows
	dCols := d.cols
	if fromRow < 0 || fromRow >= dRows || fromCol < 0 || fromCol >= dCols ||
		toRow > dRows || toCol > dCols || toRow < fromRow || toCol < fromCol {
		panic("mat: parameters are invalid or incompatible with the matrix dimensions")
	}

	y := densePool[T]().Get(toRow-fromRow, toCol-fromCol)

	if fromCol == 0 && toCol == dCols {
		copy(y.data, d.data[fromRow*dCols:toRow*dCols])
		return y
	}

	dData := d.data
	yData := y.data[:0] // exploiting append in loop
	for r := fromRow; r < toRow; r++ {
		offset := r * dCols
		yData = append(yData, dData[offset+fromCol:offset+toCol]...)
	}
	y.data = yData

	return y
}

// Reshape returns a copy of the matrix.
// It panics if the dimensions are incompatible.
func (d *Dense[T]) Reshape(rows, cols int) Matrix {
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
func (d *Dense[T]) ReshapeInPlace(rows, cols int) Matrix {
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

// Flatten creates a new row vector (1×size) corresponding to the
// "flattened" row-major ordered representation of the initial matrix.
func (d *Dense[T]) Flatten() Matrix {
	out := densePool[T]().Get(1, len(d.data))
	copy(out.data, d.data)
	return out
}

// FlattenInPlace transforms the matrix in place, changing its dimensions,
// obtaining a row vector (1×size) containing the "flattened" row-major
// ordered representation of the initial value.
// It returns the matrix itself.
func (d *Dense[T]) FlattenInPlace() Matrix {
	d.rows = 1
	d.cols = len(d.data)
	return d
}

// ResizeVector returns a resized copy of the vector.
//
// If the new size is smaller than the input vector, the remaining tail
// elements are removed. If it's bigger, the additional tail elements
// are set to zero.
func (d *Dense[T]) ResizeVector(newSize int) Matrix {
	if !(IsVector(d)) {
		panic("mat: expected vector")
	}
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
func (d *Dense[T]) T() Matrix {
	dRows := d.rows
	dCols := d.cols

	m := densePool[T]().Get(dCols, dRows)
	if IsVector(d) {
		copy(m.data, d.data)
		return m
	}
	size := len(m.data)
	index := 0
	mData := m.data
	for _, value := range d.data {
		mData[index] = value
		index += dRows
		if index >= size {
			index -= size - 1
		}
	}
	return m
}

// TransposeInPlace transposes the matrix in place, and returns the
// matrix itself.
func (d *Dense[T]) TransposeInPlace() Matrix {
	d.rows, d.cols = d.cols, d.rows

	// Vector, scalar, or empty data
	if IsVector(d) || len(d.data) <= 1 {
		return d
	}

	data := d.data

	// Square matrix
	if d.rows == d.cols {
		n := d.rows
		n1 := n - 1
		for i := 0; i < n1; i++ {
			for j := i + 1; j < n; j++ {
				k := i*n + j
				l := j*n + i
				data[k], data[l] = data[l], data[k]
			}
		}
		return d
	}

	// Rectangular matrix
	rows := d.rows
	cols := d.cols
	size := len(data)

mainLoop:
	for i := 1; i < size; i++ {
		for j := i; ; {
			j = (j%cols)*rows + j/cols
			if j == i {
				break
			}
			if j < i {
				continue mainLoop
			}
		}

		vi := data[i]
		for j := i; ; {
			k := (j%cols)*rows + j/cols
			if k == i {
				data[j] = vi
			} else {
				data[j] = data[k]
			}
			if k <= i {
				break
			}
			j = k
		}
	}

	return d
}

// Add returns the addition between the receiver and another matrix.
func (d *Dense[T]) Add(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}
	out := NewEmptyDense[T](d.rows, d.cols)
	switch any(T(0)).(type) {
	case float32:
		otherData := float32Data(other)
		matfuncs.Add32(any(d.data).([]float32), otherData, any(out.data).([]float32))
	case float64:
		otherData := float64Data(other)
		matfuncs.Add64(any(d.data).([]float64), otherData, any(out.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// AddInPlace performs the in-place addition with the other matrix.
func (d *Dense[T]) AddInPlace(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}
	switch any(T(0)).(type) {
	case float32:
		otherData := float32Data(other)
		asm32.AxpyUnitary(1, otherData, any(d.data).([]float32))
	case float64:
		otherData := float64Data(other)
		asm64.AxpyUnitary(1, otherData, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// AddScalar performs the addition between the matrix and the given value.
func (d *Dense[T]) AddScalar(n float64) Matrix {
	out := NewDense(d.rows, d.cols, d.data)
	switch any(T(0)).(type) {
	case float32:
		f32.AddConst(float32(n), any(out.data).([]float32))
	case float64:
		asm64.AddConst(n, any(out.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// AddScalarInPlace adds the scalar to all values of the matrix.
func (d *Dense[T]) AddScalarInPlace(n float64) Matrix {
	switch any(T(0)).(type) {
	case float32:
		f32.AddConst(float32(n), any(d.data).([]float32))
	case float64:
		asm64.AddConst(n, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// Sub returns the subtraction of the other matrix from the receiver.
func (d *Dense[T]) Sub(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}
	out := NewEmptyDense[T](d.rows, d.cols)
	switch any(T(0)).(type) {
	case float32:
		otherData := float32Data(other)
		asm32.AxpyUnitaryTo(any(out.data).([]float32), -1, otherData, any(d.data).([]float32))
	case float64:
		otherData := float64Data(other)
		asm64.AxpyUnitaryTo(any(out.data).([]float64), -1, otherData, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// SubInPlace performs the in-place subtraction with the other matrix.
func (d *Dense[T]) SubInPlace(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}
	switch any(T(0)).(type) {
	case float32:
		otherData := float32Data(other)
		asm32.AxpyUnitary(-1, otherData, any(d.data).([]float32))
	case float64:
		otherData := float64Data(other)
		asm64.AxpyUnitary(-1, otherData, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// SubScalar performs a subtraction between the matrix and the given value.
func (d *Dense[T]) SubScalar(n float64) Matrix {
	out := NewDense(d.rows, d.cols, d.data)
	switch any(T(0)).(type) {
	case float32:
		f32.AddConst(-float32(n), any(out.data).([]float32))
	case float64:
		asm64.AddConst(-n, any(out.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// SubScalarInPlace subtracts the scalar from the receiver's values.
func (d *Dense[T]) SubScalarInPlace(n float64) Matrix {
	switch any(T(0)).(type) {
	case float32:
		f32.AddConst(-float32(n), any(d.data).([]float32))
	case float64:
		asm64.AddConst(-n, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// Prod performs the element-wise product between the receiver and the other matrix.
func (d *Dense[T]) Prod(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}

	out := densePool[T]().Get(d.rows, d.cols)

	// Avoid bounds checks in loop
	dData := d.data
	oData := Data[T](other)
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
func (d *Dense[T]) ProdInPlace(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}
	dData := d.data
	if len(dData) == 0 {
		return d
	}
	oData := Data[T](other)
	_ = dData[len(oData)-1]
	for i, val := range oData {
		dData[i] *= val
	}
	return d
}

// ProdScalar returns the multiplication between the matrix and the given value.
func (d *Dense[T]) ProdScalar(n float64) Matrix {
	out := NewEmptyDense[T](d.rows, d.cols)
	switch any(T(0)).(type) {
	case float32:
		asm32.ScalUnitaryTo(any(out.data).([]float32), float32(n), any(d.data).([]float32))
	case float64:
		asm64.ScalUnitaryTo(any(out.data).([]float64), n, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// ProdScalarInPlace performs the in-place multiplication between the
// matrix and the given value.
func (d *Dense[T]) ProdScalarInPlace(n float64) Matrix {
	switch any(T(0)).(type) {
	case float32:
		asm32.ScalUnitary(float32(n), any(d.data).([]float32))
	case float64:
		asm64.ScalUnitary(n, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// ProdMatrixScalarInPlace multiplies the given matrix with the value,
// storing the result in the receiver.
func (d *Dense[T]) ProdMatrixScalarInPlace(m Matrix, n float64) Matrix {
	if !SameDims(d, m) {
		panic("mat: matrices have incompatible dimensions")
	}
	switch any(T(0)).(type) {
	case float32:
		mData := float32Data(m)
		asm32.ScalUnitaryTo(any(d.data).([]float32), float32(n), mData)
	case float64:
		mData := float64Data(m)
		asm64.ScalUnitaryTo(any(d.data).([]float64), n, mData)
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// Div returns the result of the element-wise division of the receiver by the other matrix.
func (d *Dense[T]) Div(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}
	out := NewEmptyDense[T](d.rows, d.cols)
	switch any(T(0)).(type) {
	case float32:
		otherData := float32Data(other)
		f32.DivTo(any(out.data).([]float32), any(d.data).([]float32), otherData)
	case float64:
		otherData := float64Data(other)
		asm64.DivTo(any(out.data).([]float64), any(d.data).([]float64), otherData)
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// DivInPlace performs the in-place element-wise division of the receiver by the other matrix.
func (d *Dense[T]) DivInPlace(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}
	dData := d.data
	if len(dData) == 0 {
		return d
	}
	oData := Data[T](other)
	_ = dData[len(oData)-1]
	for i, val := range oData {
		dData[i] *= 1.0 / val
	}
	return d
}

// Mul performs the multiplication row by column.
// If A is an i×j Matrix, and B is j×k, then the resulting Matrix
// C = AB will be i×k.
func (d *Dense[T]) Mul(other Matrix) Matrix {
	if d.cols != other.Rows() {
		panic("mat: matrices have incompatible dimensions")
	}
	outRows := d.rows
	outCols := other.Columns()

	switch any(T(0)).(type) {
	case float32:
		otherData := float32Data(other)
		if outCols != 1 {
			out := densePoolFloat32.GetEmpty(outRows, outCols)
			f32.MatrixMul(
				d.rows,                    // aRows
				d.cols,                    // aCols
				other.Columns(),           // bCols
				any(d.data).([]float32),   // a
				otherData,                 // b
				any(out.data).([]float32), // c
			)
			return out
		}

		out := densePoolFloat32.Get(outRows, outCols)
		dData := any(d.data).([]float32)
		outData := any(out.data).([]float32)

		dCols := d.cols
		from := 0
		for i := range outData {
			to := from + dCols
			outData[i] = matfuncs.DotProd32(dData[from:to], otherData)
			from = to
		}
		return out
	case float64:
		out := densePoolFloat64.GetEmpty(outRows, outCols)
		otherData := float64Data(other)
		if outCols != 1 {
			f64.MatrixMul(
				d.rows,                    // aRows
				d.cols,                    // aCols
				other.Columns(),           // bCols
				any(d.data).([]float64),   // a
				otherData,                 // b
				any(out.data).([]float64), // c
			)
			return out
		}

		asm64.GemvN(
			uintptr(d.rows),           // m
			uintptr(d.cols),           // n
			1,                         // alpha
			any(d.data).([]float64),   // a
			uintptr(d.cols),           // lda
			otherData,                 // x
			1,                         // incX
			0,                         // beta
			any(out.data).([]float64), // y
			1,                         // incY
		)
		return out
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// MulT performs the matrix multiplication row by column.
// ATB = C, where AT is the transpose of A
// if A is an r x c Matrix, and B is j x k, r = j the resulting
// Matrix C will be c x k.
func (d *Dense[T]) MulT(other Matrix) Matrix {
	if d.rows != other.Rows() {
		panic("mat: matrices have incompatible dimensions")
	}
	if other.Columns() != 1 {
		panic("mat: the other matrix must have exactly 1 column")
	}

	switch any(T(0)).(type) {
	case float32:
		out := densePoolFloat32.GetEmpty(d.cols, other.Columns())
		otherData := float32Data(other)

		dCols := d.cols
		dData := any(d.data).([]float32)
		outData := any(out.data).([]float32)

		from := 0
		for _, otherVal := range otherData {
			to := from + dCols
			asm32.AxpyUnitaryTo(outData, otherVal, dData[from:to], outData)
			from = to
		}
		return out
	case float64:
		out := densePoolFloat64.GetEmpty(d.cols, other.Columns())
		otherData := float64Data(other)

		dCols := d.cols
		dData := any(d.data).([]float64)
		outData := any(out.data).([]float64)

		from := 0
		for _, otherVal := range otherData {
			to := from + dCols
			asm64.AxpyUnitaryTo(outData, otherVal, dData[from:to], outData)
			from = to
		}
		return out
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// DotUnitary returns the dot product of two vectors as a scalar Matrix.
func (d *Dense[T]) DotUnitary(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}
	switch any(T(0)).(type) {
	case float32:
		otherData := float32Data(other)
		return NewScalar(matfuncs.DotProd32(any(d.data).([]float32), otherData))
	case float64:
		otherData := float64Data(other)
		return NewScalar(matfuncs.DotProd64(any(d.data).([]float64), otherData))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// ClipInPlace clips in place each value of the matrix.
func (d *Dense[T]) ClipInPlace(min, max float64) Matrix {
	if max < min {
		panic("mat: cannot clip values with max < min")
	}

	tMin := T(min)
	tMax := T(max)

	data := d.data
	for i, v := range data {
		switch {
		case v < tMin:
			data[i] = tMin
		case v > tMax:
			data[i] = tMax
		default:
			continue
		}
	}
	return d
}

// Maximum returns a new matrix containing the element-wise maxima.
func (d *Dense[T]) Maximum(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}
	out := densePool[T]().Get(d.rows, d.cols)
	dData := d.data
	if len(dData) == 0 {
		return out
	}
	otherData := Data[T](other)
	outData := out.data
	_ = dData[len(outData)-1]
	_ = otherData[len(outData)-1]
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
func (d *Dense[T]) Minimum(other Matrix) Matrix {
	if !SameDims(d, other) {
		panic("mat: matrices have incompatible dimensions")
	}
	out := densePool[T]().Get(d.rows, d.cols)
	dData := d.data
	if len(dData) == 0 {
		return out
	}
	otherData := Data[T](other)
	outData := out.data
	_ = dData[len(outData)-1]
	_ = otherData[len(outData)-1]
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

// Abs returns a new matrix applying the absolute value function to all elements.
func (d *Dense[T]) Abs() Matrix {
	out := densePool[T]().Get(d.rows, d.cols)
	dData := d.data
	if len(dData) == 0 {
		return out
	}
	outData := out.data
	_ = outData[len(dData)-1]
	for i, val := range dData {
		outData[i] = Abs(val)
	}
	return out
}

// Pow returns a new matrix, applying the power function with given exponent
// to all elements of the matrix.
func (d *Dense[T]) Pow(power float64) Matrix {
	out := densePool[T]().Get(d.rows, d.cols)
	dData := d.data
	if len(dData) == 0 {
		return out
	}
	outData := out.data
	_ = outData[len(dData)-1]
	for i, val := range dData {
		outData[i] = T(math.Pow(float64(val), power))
	}
	return out
}

// Sqrt returns a new matrix applying the square root function to all elements.
func (d *Dense[T]) Sqrt() Matrix {
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

// Log returns a new matrix applying the natural logarithm function to each element.
func (d *Dense[T]) Log() Matrix {
	out := densePool[T]().Get(d.rows, d.cols)
	outData := out.data
	if len(outData) == 0 {
		return out
	}
	inData := d.data
	_ = outData[len(inData)-1]
	for i, val := range inData {
		outData[i] = T(math.Log(float64(val)))
	}
	return out
}

// Exp returns a new matrix applying the base-e exponential function to each element.
func (d *Dense[T]) Exp() Matrix {
	out := densePool[T]().Get(d.rows, d.cols)
	outData := out.data
	if len(outData) == 0 {
		return out
	}
	inData := d.data
	switch any(T(0)).(type) {
	case float32:
		matfuncs.Exp32(any(inData).([]float32), any(outData).([]float32))
	case float64:
		matfuncs.Exp64(any(inData).([]float64), any(outData).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// Sigmoid returns a new matrix applying the sigmoid function to each element.
func (d *Dense[T]) Sigmoid() Matrix {
	if d.Size() == 0 {
		return d.Clone()
	}

	out := d.ProdScalar(-1).(*Dense[T])

	switch any(T(0)).(type) {
	case float32:
		matfuncs.Exp32(any(out.data).([]float32), any(out.data).([]float32))
	case float64:
		matfuncs.Exp64(any(out.data).([]float64), any(out.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}

	outData := out.data
	for i, val := range outData {
		outData[i] = 1 / (1 + val)
	}
	return out
}

// Sum returns the sum of all values of the matrix as a scalar Matrix.
func (d *Dense[T]) Sum() Matrix {
	return NewScalar(d.sum())
}

func (d *Dense[T]) sum() T {
	switch any(T(0)).(type) {
	case float32:
		return T(asm32.Sum(any(d.data).([]float32)))
	case float64:
		return T(asm64.Sum(any(d.data).([]float64)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Max returns the maximum value of the matrix as a scalar Matrix.
func (d *Dense[T]) Max() Matrix {
	return NewScalar(d.max())
}

func (d *Dense[T]) max() T {
	if len(d.data) == 0 {
		panic("mat: cannot find the maximum value from an empty matrix")
	}
	max := d.data[0]
	for _, v := range d.data[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// Min returns the minimum value of the matrix as a scalar Matrix.
func (d *Dense[T]) Min() Matrix {
	if len(d.data) == 0 {
		panic("mat: cannot find the minimum value in an empty matrix")
	}
	min := d.data[0]
	for _, v := range d.data[1:] {
		if v < min {
			min = v
		}
	}
	return NewScalar(min)
}

// ArgMax returns the index of the vector's element with the maximum value.
func (d *Dense[T]) ArgMax() int {
	if !IsVector(d) {
		panic("mat: expected vector")
	}
	data := d.data
	if len(data) == 0 {
		panic("mat: cannot find arg-max from an empty vector")
	}
	maxIndex := 0
	maxValue := data[0]
	for i, v := range data {
		if v > maxValue {
			maxIndex = i
			maxValue = v
		}
	}
	return maxIndex
}

// Softmax applies the softmax function to the vector, returning the
// result as a new column vector.
func (d *Dense[T]) Softmax() Matrix {
	if !IsVector(d) {
		panic("mat: expected vector")
	}

	if d.Size() == 0 {
		return d.NewEmptyVec(0)
	}

	max := float64(d.max())

	out := d.SubScalar(max).(*Dense[T])
	if out.cols != 1 {
		out.TransposeInPlace()
	}

	switch any(T(0)).(type) {
	case float32:
		outData := any(out.data).([]float32)
		matfuncs.Exp32(outData, outData)
	case float64:
		outData := any(out.data).([]float64)
		matfuncs.Exp64(outData, outData)
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}

	sum := out.sum()
	out.ProdScalarInPlace(float64(1 / sum))

	return out
}

// CumSum computes the cumulative sum of the vector's elements, returning
// the result as a new column vector.
func (d *Dense[T]) CumSum() Matrix {
	if !IsVector(d) {
		panic("mat: expected vector")
	}

	out := densePool[T]().Get(len(d.data), 1)
	if len(d.data) == 0 {
		return out
	}

	switch any(T(0)).(type) {
	case float32:
		f32.CumSum(any(out.data).([]float32), any(d.data).([]float32))
	case float64:
		asm64.CumSum(any(out.data).([]float64), any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}

	return out
}

// Range creates a new vector initialized with data extracted from the
// matrix raw data, from start (inclusive) to end (exclusive).
func (d *Dense[T]) Range(start, end int) Matrix {
	if !IsVector(d) {
		panic("mat: expected vector")
	}
	if end < start {
		panic("mat: cannot extract range with end < start")
	}
	if end < 0 || start < 0 {
		panic("mat: negative values for range indices are not allowed")
	}
	return NewVecDense(d.data[start:end])
}

// SplitV splits the vector in N chunks of given sizes,
// so that N[i] has size sizes[i].
func (d *Dense[T]) SplitV(sizes ...int) []Matrix {
	if !IsVector(d) {
		panic("mat: expected vector")
	}
	if len(sizes) == 0 {
		return nil
	}
	out := make([]Matrix, len(sizes))
	offset := 0
	for i, size := range sizes {
		if size < 0 {
			panic("mat: a negative size is not allowed")
		}
		startIndex := offset
		offset = startIndex + size
		if startIndex >= len(d.data) && offset > startIndex {
			panic("mat: sizes out of bounds")
		}
		out[i] = NewVecDense(d.data[startIndex:offset])
	}
	return out
}

// Augment places the identity matrix at the end of the original matrix.
func (d *Dense[T]) Augment() Matrix {
	if d.cols != d.rows {
		panic("mat: matrix must be square")
	}
	// TODO: rewrite for better performance
	out := NewEmptyDense[T](d.rows, d.cols*2)
	for i := 0; i < d.rows; i++ {
		for j := 0; j < d.cols; j++ {
			out.SetScalar(i, j, d.ScalarAt(i, j))
		}
		out.set(i, i+d.rows, 1.0)
	}
	return out
}

// SwapInPlace swaps two rows of the matrix in place.
func (d *Dense[T]) SwapInPlace(r1, r2 int) Matrix {
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
func (d *Dense[T]) PadRows(n int) Matrix {
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
func (d *Dense[T]) PadColumns(n int) Matrix {
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

// AppendRows returns a copy of the matrix with len(vs) additional tail rows,
// being each new row filled with the values of each given vector.
//
// It accepts row or column vectors indifferently, virtually treating all of
// them as row vectors.
func (d *Dense[T]) AppendRows(vs ...Matrix) Matrix {
	cols := d.cols
	out := densePool[T]().Get(d.rows+len(vs), cols)
	dData := d.data
	outData := out.data
	copy(outData[:len(dData)], dData)

	offset := len(dData)
	for _, v := range vs {
		if !IsVector(v) || v.Size() != cols {
			panic("mat: expected vectors with same size of matrix columns")
		}
		vData := Data[T](v)
		end := offset + cols
		copy(outData[offset:end], vData)
		offset = end
	}

	return out
}

// Norm returns the vector's norm. Use pow = 2.0 to compute the Euclidean norm.
// The result is a scalar Matrix.
func (d *Dense[T]) Norm(pow float64) Matrix {
	return NewScalar(T(d.norm(pow)))
}

func (d *Dense[T]) norm(pow float64) float64 {
	if !IsVector(d) {
		panic("mat: expected vector")
	}
	var s float64
	for _, x := range d.data {
		s += math.Pow(float64(x), pow)
	}
	return math.Pow(s, 1/pow)
}

// Normalize2 normalizes an array with the Euclidean norm.
func (d *Dense[T]) Normalize2() Matrix {
	norm2 := d.norm(2)
	if norm2 == 0 {
		return d.Clone()
	}
	return d.ProdScalar(1 / norm2)
}

// Pivoting returns the partial pivots of a square matrix to reorder rows.
// Considerate square sub-matrix from element (offset, offset).
func (d *Dense[T]) Pivoting(row int) (Matrix, bool, [2]int) {
	if d.rows != d.cols {
		panic("mat: matrix must be square")
	}
	if row < 0 || row >= d.rows {
		panic("mat: row out of bounds")
	}

	pv := make([]int, d.cols)
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

	var positions [2]int
	swap := j != row
	if swap {
		pv[row], pv[j] = pv[j], pv[row]
		positions = [2]int{row, j}
	}

	p := NewEmptyDense[T](d.cols, d.cols)
	for r, c := range pv {
		p.data[r*d.cols+c] = 1
	}
	return p, swap, positions
}

// LU performs lower–upper (LU) decomposition of a square matrix D such as
// PLU = D, L is lower diagonal and U is upper diagonal, p are pivots.
func (d *Dense[T]) LU() (l, u, p Matrix) {
	if d.rows != d.cols {
		panic("mat: matrix must be square")
	}
	u = NewDense(d.rows, d.cols, d.data)
	p = NewIdentityDense[T](d.cols)
	l = NewEmptyDense[T](d.cols, d.cols)
	lData := Data[T](l)
	for i := 0; i < d.cols; i++ {
		_, swap, positions := u.Pivoting(i)
		if swap {
			u.SwapInPlace(positions[0], positions[1])
			p.SwapInPlace(positions[0], positions[1])
			l.SwapInPlace(positions[0], positions[1])
		}
		lt := NewIdentityDense[T](d.cols)
		ltData := lt.data
		uData := Data[T](u)
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
func (d *Dense[T]) Inverse() Matrix {
	if d.cols != d.rows {
		panic("mat: matrix must be square")
	}
	out := NewEmptyDense[T](d.cols, d.cols)
	outData := out.data
	s := NewEmptyDense[T](d.cols, d.cols)
	sData := s.data
	l, u, p := d.LU()
	lData := Data[T](l)
	uData := Data[T](u)
	pData := Data[T](p)
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

// VecForEach calls fn for each element of the vector.
// It panics if the receiver is not a vector.
func (d *Dense[T]) VecForEach(fn func(i int, v float64)) {
	if !IsVector(d) {
		panic("mat: expected vector")
	}
	for i, v := range d.data {
		fn(i, float64(v))
	}
}

// Apply creates a new matrix executing the unary function fn.
func (d *Dense[T]) Apply(fn func(r, c int, v float64) float64) Matrix {
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
		outData[i] = T(fn(r, c, float64(v)))
		c++
		if c == d.cols {
			r++
			c = 0
		}
	}

	return out
}

// ApplyInPlace executes the unary function fn over the matrix a,
// and stores the result in the receiver, returning the receiver itself.
func (d *Dense[T]) ApplyInPlace(fn func(r, c int, v float64) float64, a Matrix) Matrix {
	if !SameDims(d, a) {
		panic("mat: incompatible matrix dimensions")
	}
	aData := Data[T](a)
	lastIndex := len(aData) - 1
	if lastIndex < 0 {
		return d
	}
	r := 0
	c := 0
	dData := d.data
	_ = dData[lastIndex]
	for i, val := range aData {
		dData[i] = T(fn(r, c, float64(val)))
		c++
		if c == d.cols {
			r++
			c = 0
		}
	}
	return d
}

// ApplyWithAlpha creates a new matrix executing the unary function fn,
// taking additional parameters alpha.
func (d *Dense[T]) ApplyWithAlpha(fn func(r, c int, v float64, alpha ...float64) float64, alpha ...float64) Matrix {
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
		outData[i] = T(fn(r, c, float64(v), alpha...))
		c++
		if c == d.cols {
			r++
			c = 0
		}
	}

	return out
}

// ApplyWithAlphaInPlace executes the unary function fn over the matrix a,
// taking additional parameters alpha, and stores the result in the
// receiver, returning the receiver itself.
func (d *Dense[T]) ApplyWithAlphaInPlace(fn func(r, c int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64) Matrix {
	if !SameDims(d, a) {
		panic("mat: incompatible matrix dimensions")
	}
	// TODO: rewrite for better performance
	for r := 0; r < d.rows; r++ {
		for c := 0; c < d.cols; c++ {
			d.data[r*d.cols+c] = T(fn(r, c, a.ScalarAt(r, c).F64(), alpha...))
		}
	}
	return d
}

// DoNonZero calls a function for each non-zero element of the matrix.
// The parameters of the function are the element's indices and value.
func (d *Dense[T]) DoNonZero(fn func(r, c int, v float64)) {
	for r, di := 0, 0; r < d.rows; r++ {
		for c := 0; c < d.cols; c, di = c+1, di+1 {
			v := d.data[di]
			if v == 0 {
				continue
			}
			fn(r, c, float64(v))
		}
	}
}

// DoVecNonZero calls a function for each non-zero element of the vector.
// The parameters of the function are the element's index and value.
func (d *Dense[T]) DoVecNonZero(fn func(i int, v float64)) {
	if !IsVector(d) {
		panic("mat: expected vector")
	}
	for i, v := range d.data {
		if v == 0 {
			continue
		}
		fn(i, float64(v))
	}
}

// Clone returns a new matrix, copying all its values from the receiver.
func (d *Dense[T]) Clone() Matrix {
	out := densePool[T]().Get(d.rows, d.cols)
	copy(out.data, d.data)
	return out
}

// Copy copies the data from the other matrix to the receiver.
// It panics if the matrices have different dimensions.
func (d *Dense[T]) Copy(other Matrix) {
	if !SameDims(d, other) {
		panic("mat: incompatible matrix dimensions")
	}
	copy(d.data, Data[T](other))
}

// String returns a string representation of the matrix.
func (d *Dense[T]) String() string {
	return fmt.Sprintf("Matrix|Dense[%T](%d×%d)%v", T(0), d.rows, d.cols, d.data)
}

// NewMatrix creates a new matrix, of the same type of the receiver, of
// size rows×cols, initialized with a copy of raw data.
//
// Rows and columns MUST not be negative, and the length of data MUST be
// equal to rows*cols, otherwise the method panics.
func (d *Dense[T]) NewMatrix(rows, cols int, data float.Slice) Matrix {
	return NewDense[T](rows, cols, float.SliceValueOf[T](data))
}

// NewVec creates a new column vector (len(data)×1), of the same type of
// the receiver, initialized with a copy of raw data.
func (d *Dense[T]) NewVec(data float.Slice) Matrix {
	return NewVecDense[T](float.SliceValueOf[T](data))
}

// NewScalar creates a new 1×1 matrix, of the same type of the receiver,
// containing the given value.
func (d *Dense[T]) NewScalar(v float64) Matrix {
	return NewScalar(T(v))
}

// NewEmptyVec creates a new vector, of the same type of the receiver,
// with dimensions size×1, initialized with zeros.
func (d *Dense[T]) NewEmptyVec(size int) Matrix {
	return NewEmptyVecDense[T](size)
}

// NewEmptyMatrix creates a new rows×cols matrix, of the same type of the
// receiver, initialized with zeros.
func (d *Dense[T]) NewEmptyMatrix(rows, cols int) Matrix {
	return NewEmptyDense[T](rows, cols)
}

// NewInitMatrix creates a new rows×cols dense matrix, of the same type
// of the receiver, initialized with a constant value.
func (d *Dense[T]) NewInitMatrix(rows, cols int, v float64) Matrix {
	return NewInitDense(rows, cols, T(v))
}

// NewInitFuncMatrix creates a new rows×cols dense matrix, of the same type
// of the receiver, initialized with the values returned from the
// callback function.
func (d *Dense[T]) NewInitFuncMatrix(rows, cols int, fn func(r, c int) float64) Matrix {
	return NewInitFuncDense(rows, cols, func(r, c int) T {
		return T(fn(r, c))
	})
}

// NewInitVec creates a new column vector (size×1), of the same type of
// the receiver, initialized with a constant value.
func (d *Dense[T]) NewInitVec(size int, v float64) Matrix {
	return NewInitVecDense(size, T(v))
}

// NewIdentityMatrix creates a new square identity matrix (size×size), of
// the same type of the receiver, that is, with ones on the diagonal
// and zeros elsewhere.
func (d *Dense[T]) NewIdentityMatrix(size int) Matrix {
	return NewIdentityDense[T](size)
}

// NewOneHotVec creates a new one-hot column vector (size×1), of the same
// type of the receiver.
func (d *Dense[T]) NewOneHotVec(size int, oneAt int) Matrix {
	return NewOneHotVecDense[T](size, oneAt)
}

// NewConcatV creates a new column vector, of the same type of the receiver,
// concatenating two or more vectors "vertically"
// It accepts row or column vectors indifferently, virtually
// treating all of them as column vectors.
func (d *Dense[T]) NewConcatV(vs ...Matrix) Matrix {
	return ConcatV[T](vs...)
}

// NewStack creates a new matrix, of the same type of the receiver, stacking
// two or more vectors of the same size on top of each other; the result is
// a new matrix where each row contains the data of each input vector.
// It accepts row or column vectors indifferently, virtually treating all of
// them as row vectors.
func (d *Dense[T]) NewStack(vs ...Matrix) Matrix {
	return Stack[T](vs...)
}
