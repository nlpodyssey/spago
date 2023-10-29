// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"log"
	"math"
	"sync"

	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/internal/f32"
	"github.com/nlpodyssey/spago/mat/internal/f32/asm32"
	"github.com/nlpodyssey/spago/mat/internal/f64"
	"github.com/nlpodyssey/spago/mat/internal/f64/asm64"
	"github.com/nlpodyssey/spago/mat/internal/matfuncs"
)

// A Dense matrix implementation.
type Dense[T float.DType] struct {
	gradMu       sync.RWMutex
	data         []T
	grad         *Dense[T]
	shape        []int
	requiresGrad bool // default: false
}

// makeDense returns a Dense matrix.
func makeDense[T float.DType](array []T, shape ...int) *Dense[T] {
	if len(array) != calculateSize(shape) {
		log.Fatalf("mat: incompatible size, expected %d, actual %d", calculateSize(shape), len(array))
	}
	return &Dense[T]{
		shape: shape,
		data:  array,
	}
}

func malloc[T float.DType](size int) []T {
	return make([]T, size)
}

// Shape returns the size in each dimension.
func (d *Dense[_]) Shape() []int {
	return d.shape
}

// Dims returns the number of dimensions.
func (d *Dense[_]) Dims() int {
	return 2 // rows and columns
}

// The Size of the matrix (rows*columns).
func (d *Dense[_]) Size() int {
	return len(d.data)
}

// Data returns the underlying data of the matrix, as a raw one-dimensional
// slice of values in row-major order.
func (d *Dense[T]) Data() float.Slice {
	return float.Make(d.data...)
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
	return NewDense[T](WithShape(d.shape...))
}

// OnesLike returns a new matrix with the same dimensions of the
// receiver, initialized with ones.
func (d *Dense[T]) OnesLike() Matrix {
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
	data := out.data // avoid bounds check in loop
	for i := range data {
		data[i] = 1.0
	}
	return out
}

// Scalar returns the scalar value.
// It panics if the matrix does not contain exactly one element.
func (d *Dense[T]) Item() float.Float {
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

// SetAt sets the value m at the given indices.
// It panics if the given indices are out of range.
func (d *Dense[T]) SetAt(m Tensor, indices ...int) {
	d.set(float.ValueOf[T](m.Item()), indices...)
}

// At returns the value at the given indices.
// It panics if the given indices are out of range.
func (d *Dense[T]) At(i ...int) Tensor {
	return Scalar[T](d.at(i...))
}

// SetScalar sets the value v at the given indices.
// It panics if the given indices are out of range.
func (d *Dense[T]) SetScalar(v float.Float, indices ...int) {
	d.set(float.ValueOf[T](v), indices...)
}

// ScalarAt returns the value at the given indices.
// It panics if the given indices are out of range.
func (d *Dense[T]) ScalarAt(indices ...int) float.Float {
	return float.Interface(d.at(indices...))
}

func (d *Dense[T]) set(v T, i ...int) {
	switch len(i) {
	case 1:
		if d.shape[0] != 1 && d.shape[1] != 1 {
			panic("Dense structure is not a 1-dimensional array")
		}
		idx := i[0]
		if idx < 0 || idx >= len(d.data) {
			panic("Index 'i' out of range")
		}
		d.data[idx] = v
	case 2:
		r, c := i[0], i[1]
		if r < 0 || r >= d.shape[0] {
			panic("Row index 'r' out of range")
		}
		if c < 0 || c >= d.shape[1] {
			panic("Column index 'c' out of range")
		}
		d.data[r*d.shape[1]+c] = v
	default:
		panic("Incorrect number of indices provided")
	}
}

func (d *Dense[T]) at(i ...int) T {
	switch len(i) {
	case 1:
		if d.shape[0] != 1 && d.shape[1] != 1 {
			panic("Dense structure is not a 1-dimensional array")
		}
		idx := i[0]
		if idx < 0 || idx >= len(d.data) {
			panic("Index 'i' out of range")
		}
		return d.data[idx]
	case 2:
		r, c := i[0], i[1]
		if r < 0 || r >= d.shape[0] {
			panic("Row index 'r' out of range")
		}
		if c < 0 || c >= d.shape[1] {
			panic("Column index 'c' out of range")
		}
		return d.data[r*d.shape[1]+c]
	default:
		panic("Incorrect number of indices provided")
	}
}

// ExtractRow returns a copy of the i-th row of the matrix,
// as a row vector (1×cols).
func (d *Dense[T]) ExtractRow(i int) Matrix {
	if i < 0 || i >= d.shape[0] {
		panic("mat: index out of range")
	}
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.shape[1]), 1, d.shape[1])
	start := i * d.shape[1]
	copy(out.data, d.data[start:start+d.shape[1]])
	return out
}

// ExtractColumn returns a copy of the i-th column of the matrix,
// as a column vector (rows×1).
func (d *Dense[T]) ExtractColumn(i int) Matrix {
	if i < 0 || i >= d.shape[1] {
		panic("mat: index out of range")
	}
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.shape[0]), d.shape[0], 1)
	dData := d.data
	outData := out.data
	for k := range outData {
		outData[k] = dData[k*d.shape[1]+i]
	}
	return out
}

// Slice returns a new matrix obtained by slicing the receiver across the
// given positions. The parameters "fromRow" and "fromCol" are inclusive,
// while "toRow" and "toCol" are exclusive.
func (d *Dense[T]) Slice(fromRow, fromCol, toRow, toCol int) Matrix {
	dRows := d.shape[0]
	dCols := d.shape[1]
	if fromRow < 0 || fromRow >= dRows || fromCol < 0 || fromCol >= dCols ||
		toRow > dRows || toCol > dCols || toRow < fromRow || toCol < fromCol {
		panic("mat: parameters are invalid or incompatible with the matrix dimensions")
	}

	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	y := makeDense[T](malloc[T]((toRow-fromRow)*(toCol-fromCol)), toRow-fromRow, toCol-fromCol)

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
func (d *Dense[T]) Reshape(shape ...int) Matrix {
	if len(shape) != 2 {
		panic("mat: reshape requires two dimensions")
	}
	rows, cols := shape[0], shape[1]
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	if rows*cols != len(d.data) {
		panic(fmt.Sprintf("mat: wrong matrix dimensions. Size (rows*cols) must be: %d", len(d.data)))
	}

	return NewDense[T](WithShape(rows, cols), WithBacking(copySlice(d.data)))
}

func copySlice[T float.DType](src []T) []T {
	dst := make([]T, len(src))
	copy(dst, src)
	return dst
}

// ReshapeInPlace changes the dimensions of the matrix in place and returns the
// matrix itself.
// It panics if the dimensions are incompatible.
func (d *Dense[T]) ReshapeInPlace(shape ...int) Matrix {
	if len(shape) != 2 {
		panic("mat: reshape requires two dimensions")
	}
	rows, cols := shape[0], shape[1]
	if rows < 0 || cols < 0 {
		panic("mat: negative values for rows and cols are not allowed")
	}
	if rows*cols != len(d.data) {
		panic(fmt.Sprintf("mat: wrong matrix dimensions. Size (rows*cols) must be: %d", len(d.data)))
	}
	d.shape[0] = rows
	d.shape[1] = cols
	return d
}

// Flatten creates a new row vector (1×size) corresponding to the
// "flattened" row-major ordered representation of the initial matrix.
func (d *Dense[T]) Flatten() Matrix {
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](len(d.data)), 1, len(d.data))
	copy(out.data, d.data)
	return out
}

// FlattenInPlace transforms the matrix in place, changing its dimensions,
// obtaining a row vector (1×size) containing the "flattened" row-major
// ordered representation of the initial value.
// It returns the matrix itself.
func (d *Dense[T]) FlattenInPlace() Matrix {
	d.shape[0] = 1
	d.shape[1] = len(d.data)
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
		return NewDense[T](WithBacking(copySlice(d.data[:newSize])))
	}

	y := NewDense[T](WithShape(newSize))
	copy(y.data[:dSize], d.data)
	return y
}

// T returns the transpose of the matrix.
func (d *Dense[T]) T() Matrix {
	dRows := d.shape[0]
	dCols := d.shape[1]

	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	m := makeDense[T](malloc[T](dCols*dRows), dCols, dRows)
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
	d.shape[0], d.shape[1] = d.shape[1], d.shape[0]

	// Vector, scalar, or empty data
	if IsVector(d) || len(d.data) <= 1 {
		return d
	}

	data := d.data

	// Square matrix
	if d.shape[0] == d.shape[1] {
		n := d.shape[0]
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
	rows := d.shape[0]
	cols := d.shape[1]
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
	out := NewDense[T](WithShape(d.shape...))
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
		dData := any(d.data).([]float32)
		otherData := float32Data(other)
		matfuncs.Add32(dData, otherData, dData)
	case float64:
		dData := any(d.data).([]float64)
		otherData := float64Data(other)
		matfuncs.Add64(dData, otherData, dData)
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// AddScalar performs the addition between the matrix and the given value.
func (d *Dense[T]) AddScalar(n float64) Matrix {
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
	switch any(T(0)).(type) {
	case float32:
		matfuncs.AddConst32(float32(n), any(d.data).([]float32), any(out.data).([]float32))
	case float64:
		matfuncs.AddConst64(n, any(d.data).([]float64), any(out.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// AddScalarInPlace adds the scalar to all values of the matrix.
func (d *Dense[T]) AddScalarInPlace(n float64) Matrix {
	switch any(T(0)).(type) {
	case float32:
		dData := any(d.data).([]float32)
		matfuncs.AddConst32(float32(n), dData, dData)
	case float64:
		dData := any(d.data).([]float64)
		matfuncs.AddConst64(n, dData, dData)
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
	out := NewDense[T](WithShape(d.shape...))
	switch any(T(0)).(type) {
	case float32:
		otherData := float32Data(other)
		matfuncs.Sub32(any(d.data).([]float32), otherData, any(out.data).([]float32))
	case float64:
		otherData := float64Data(other)
		matfuncs.Sub64(any(d.data).([]float64), otherData, any(out.data).([]float64))
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
		matfuncs.Sub32(any(d.data).([]float32), otherData, any(d.data).([]float32))
	case float64:
		otherData := float64Data(other)
		matfuncs.Sub64(any(d.data).([]float64), otherData, any(d.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// SubScalar performs a subtraction between the matrix and the given value.
func (d *Dense[T]) SubScalar(n float64) Matrix {
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
	switch any(T(0)).(type) {
	case float32:
		matfuncs.AddConst32(float32(-n), any(d.data).([]float32), any(out.data).([]float32))
	case float64:
		matfuncs.AddConst64(-n, any(d.data).([]float64), any(out.data).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// SubScalarInPlace subtracts the scalar from the receiver's values.
func (d *Dense[T]) SubScalarInPlace(n float64) Matrix {
	switch any(T(0)).(type) {
	case float32:
		dData := any(d.data).([]float32)
		matfuncs.AddConst32(float32(-n), dData, dData)
	case float64:
		dData := any(d.data).([]float64)
		matfuncs.AddConst64(-n, dData, dData)
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

	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)

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
	out := NewDense[T](WithShape(d.shape...))
	switch any(T(0)).(type) {
	case float32:
		matfuncs.MulConst32(float32(n), any(d.data).([]float32), any(out.data).([]float32))
	case float64:
		matfuncs.MulConst64(n, any(d.data).([]float64), any(out.data).([]float64))
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
		dData := any(d.data).([]float32)
		matfuncs.MulConst32(float32(n), dData, dData)
	case float64:
		dData := any(d.data).([]float64)
		matfuncs.MulConst64(n, dData, dData)
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
		matfuncs.MulConst32(float32(n), mData, any(d.data).([]float32))
	case float64:
		mData := float64Data(m)
		matfuncs.MulConst64(n, mData, any(d.data).([]float64))
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
	out := NewDense[T](WithShape(d.shape...))
	switch any(T(0)).(type) {
	case float32:
		otherData := float32Data(other)
		matfuncs.Div32(any(d.data).([]float32), otherData, any(out.data).([]float32))
	case float64:
		otherData := float64Data(other)
		matfuncs.Div64(any(d.data).([]float64), otherData, any(out.data).([]float64))
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
	switch any(T(0)).(type) {
	case float32:
		dData := any(d.data).([]float32)
		otherData := float32Data(other)
		matfuncs.Div32(dData, otherData, dData)
	case float64:
		dData := any(d.data).([]float64)
		otherData := float64Data(other)
		matfuncs.Div64(dData, otherData, dData)
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return d
}

// Mul performs the multiplication row by column.
// If A is an i×j Matrix, and B is j×k, then the resulting Matrix
// C = AB will be i×k.
func (d *Dense[T]) Mul(other Matrix) Matrix {
	otherShape := other.Shape()
	otherRows, otherCols := otherShape[0], otherShape[1]

	if d.shape[1] != otherRows {
		panic("mat: matrices have incompatible dimensions")
	}
	outRows := d.shape[0]
	outCols := otherCols

	switch any(T(0)).(type) {
	case float32:
		otherData := float32Data(other)
		if outCols != 1 {
			out := makeDense[float32](malloc[float32](outRows*outCols), outRows, outCols)
			f32.MatrixMul(
				d.shape[0],                // aRows
				d.shape[1],                // aCols
				otherCols,                 // bCols
				any(d.data).([]float32),   // a
				otherData,                 // b
				any(out.data).([]float32), // c
			)
			return out
		}

		// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
		out := makeDense[float32](malloc[float32](outRows*outCols), outRows, outCols)
		dData := any(d.data).([]float32)
		outData := any(out.data).([]float32)

		dCols := d.shape[1]
		from := 0
		for i := range outData {
			to := from + dCols
			outData[i] = matfuncs.DotProd32(dData[from:to], otherData)
			from = to
		}
		return out
	case float64:
		out := makeDense[float64](malloc[float64](outRows*outCols), outRows, outCols)
		otherData := float64Data(other)
		if outCols != 1 {
			f64.MatrixMul(
				d.shape[0],                // aRows
				d.shape[1],                // aCols
				otherCols,                 // bCols
				any(d.data).([]float64),   // a
				otherData,                 // b
				any(out.data).([]float64), // c
			)
			return out
		}

		asm64.GemvN(
			uintptr(d.shape[0]),       // m
			uintptr(d.shape[1]),       // n
			1,                         // alpha
			any(d.data).([]float64),   // a
			uintptr(d.shape[1]),       // lda
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
	otherShape := other.Shape()
	otherRows, otherCols := otherShape[0], otherShape[1]

	if d.shape[0] != otherRows {
		panic("mat: matrices have incompatible dimensions")
	}
	if otherCols != 1 {
		panic("mat: the other matrix must have exactly 1 column")
	}

	switch any(T(0)).(type) {
	case float32:
		out := makeDense[float32](malloc[float32](d.shape[1]*otherCols), d.shape[1], otherCols)
		otherData := float32Data(other)

		dCols := d.shape[1]
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
		out := makeDense[float64](malloc[float64](d.shape[1]*otherCols), d.shape[1], otherCols)
		otherData := float64Data(other)

		dCols := d.shape[1]
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
		return Scalar(matfuncs.DotProd32(any(d.data).([]float32), otherData))
	case float64:
		otherData := float64Data(other)
		return Scalar(matfuncs.DotProd64(any(d.data).([]float64), otherData))
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
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
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
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
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
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
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
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
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
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
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
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
	outData := out.data
	if len(outData) == 0 {
		return out
	}
	inData := d.data
	switch any(T(0)).(type) {
	case float32:
		matfuncs.Log32(any(inData).([]float32), any(outData).([]float32))
	case float64:
		matfuncs.Log64(any(inData).([]float64), any(outData).([]float64))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
	return out
}

// Exp returns a new matrix applying the base-e exponential function to each element.
func (d *Dense[T]) Exp() Matrix {
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
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
	return Scalar(d.sum())
}

func (d *Dense[T]) sum() T {
	switch any(T(0)).(type) {
	case float32:
		return T(matfuncs.Sum32(any(d.data).([]float32)))
	case float64:
		return T(matfuncs.Sum64(any(d.data).([]float64)))
	default:
		panic(fmt.Sprintf("mat: unexpected type %T", T(0)))
	}
}

// Max returns the maximum value of the matrix as a scalar Matrix.
func (d *Dense[T]) Max() Matrix {
	return Scalar(d.max())
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
	return Scalar(min)
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
		return d.NewMatrix(WithShape(0))
	}

	max := float64(d.max())

	out := d.SubScalar(max).(*Dense[T])
	if out.shape[1] != 1 {
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

	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](len(d.data)), len(d.data), 1)
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
	return NewDense[T](WithBacking(copySlice(d.data[start:end])))
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
		out[i] = NewDense[T](WithBacking(copySlice(d.data[startIndex:offset])))
	}
	return out
}

// Augment places the identity matrix at the end of the original matrix.
func (d *Dense[T]) Augment() Matrix {
	if d.shape[1] != d.shape[0] {
		panic("mat: matrix must be square")
	}
	// TODO: rewrite for better performance
	out := NewDense[T](WithShape(d.shape[0], d.shape[1]*2))
	for i := 0; i < d.shape[0]; i++ {
		for j := 0; j < d.shape[1]; j++ {
			out.SetScalar(d.ScalarAt(i, j), i, j)
		}
		out.set(1.0, i, i+d.shape[0])
	}
	return out
}

// SwapInPlace swaps two rows of the matrix in place.
func (d *Dense[T]) SwapInPlace(r1, r2 int) Matrix {
	if r1 < 0 || r1 >= d.shape[0] {
		panic("mat: 'r1' argument out of range")
	}
	if r2 < 0 || r2 >= d.shape[0] {
		panic("mat: 'r2' argument out of range")
	}
	// TODO: rewrite for better performance
	for j := 0; j < d.shape[1]; j++ {
		a, b := r1*d.shape[1]+j, r2*d.shape[1]+j
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
	cols := d.shape[1]
	dRows := d.shape[0]
	yRows := dRows + n
	y := NewDense[T](WithShape(yRows, cols))

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
	rows := d.shape[0]
	dCols := d.shape[1]
	yCols := dCols + n
	y := NewDense[T](WithShape(rows, yCols))

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
	cols := d.shape[1]
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T]((d.shape[0]+len(vs))*cols), d.shape[0]+len(vs), cols)
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
	return Scalar(T(d.norm(pow)))
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

// Apply creates a new matrix executing the unary function fn.
func (d *Dense[T]) Apply(fn func(r, c int, v float64) float64) Matrix {
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
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
		if c == d.shape[1] {
			r++
			c = 0
		}
	}

	return out
}

// ApplyInPlace executes the unary function fn over the matrix `a`,
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
		if c == d.shape[1] {
			r++
			c = 0
		}
	}
	return d
}

// ApplyWithAlpha creates a new matrix executing the unary function fn,
// taking additional alpha.
func (d *Dense[T]) ApplyWithAlpha(fn func(r, c int, v float64, alpha ...float64) float64, alpha ...float64) Matrix {
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
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
		if c == d.shape[1] {
			r++
			c = 0
		}
	}

	return out
}

// ApplyWithAlphaInPlace executes the unary function fn over the matrix `a`,
// taking additional parameters alpha, and stores the result in the
// receiver, returning the receiver itself.
func (d *Dense[T]) ApplyWithAlphaInPlace(fn func(r, c int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64) Matrix {
	if !SameDims(d, a) {
		panic("mat: incompatible matrix dimensions")
	}
	// TODO: rewrite for better performance
	for r := 0; r < d.shape[0]; r++ {
		for c := 0; c < d.shape[1]; c++ {
			d.data[r*d.shape[1]+c] = T(fn(r, c, a.ScalarAt(r, c).F64(), alpha...))
		}
	}
	return d
}

// DoNonZero calls a function for each non-zero element of the matrix.
// The parameters of the function are the element's indices and value.
func (d *Dense[T]) DoNonZero(fn func(r, c int, v float64)) {
	for r, di := 0, 0; r < d.shape[0]; r++ {
		for c := 0; c < d.shape[1]; c, di = c+1, di+1 {
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
	// Note: Consider that for performance optimization, it's not necessary to initialize the underlying slice to zero.
	out := makeDense[T](malloc[T](d.Size()), d.shape...)
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
	return fmt.Sprintf("Matrix|Dense[%T](%d×%d)%v", T(0), d.shape[0], d.shape[1], d.data)
}

// NewMatrix creates a new matrix, of the same type of the receiver, of
// size rows×cols, initialized with a copy of raw data.
//
// Rows and columns MUST not be negative, and the length of data MUST be
// equal to rows*cols, otherwise the method panics.
func (d *Dense[T]) NewMatrix(opts ...OptionsFunc) Matrix {
	return NewDense[T](opts...)
}

func (d *Dense[T]) NewScalar(v float64, opts ...OptionsFunc) Matrix {
	return Scalar[T](T(v), opts...)
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
