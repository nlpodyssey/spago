// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat/internal/asm/f64"
	"math"
)

var _ Matrix = &Dense{}

type Dense struct {
	rows   int
	cols   int
	size   int // rows*cols
	data   []float64
	viewOf *Dense // default nil
}

// NewDense returns a new rows x cols dense matrix populated with a copy of the elements.
// The elements cannot be nil, panic otherwise. Use NewEmptyDense to initialize an empty matrix.
func NewDense(rows, cols int, elements []float64) *Dense {
	if elements == nil {
		panic("mat: elements cannot be nil. Use NewEmptyDense() instead.")
	}
	if len(elements) != rows*cols {
		panic(fmt.Sprintf("mat: wrong matrix dimensions. Elements size must be: %d", rows*cols))
	}
	return &Dense{
		rows:   rows,
		cols:   cols,
		size:   rows * cols,
		data:   append([]float64(nil), elements...),
		viewOf: nil,
	}
}

// NewVecDense returns a new column vector populated with a copy of the elements.
// The elements cannot be nil, panic otherwise. Use NewEmptyVecDense to initialize an empty matrix.
func NewVecDense(elements []float64) *Dense {
	if elements == nil {
		panic("mat: elements cannot be nil. Use NewEmptyVecDense() instead.")
	}
	size := len(elements)
	return &Dense{
		rows:   size,
		cols:   1,
		size:   size,
		data:   append([]float64(nil), elements...),
		viewOf: nil,
	}
}

// NewScalar returns a new 1x1 matrix containing the input value.
func NewScalar(n float64) *Dense {
	data := []float64{n}
	return &Dense{
		rows:   1,
		cols:   1,
		size:   1,
		data:   data,
		viewOf: nil,
	}
}

// NewEmptyVecDense returns a new vector of the given size, initialized to zeros.
func NewEmptyVecDense(size int) *Dense {
	return &Dense{
		rows:   size,
		cols:   1,
		size:   size,
		data:   make([]float64, size),
		viewOf: nil,
	}
}

// NewEmptyVecDense returns a new rows x cols matrix initialized to zeros.
func NewEmptyDense(rows, cols int) *Dense {
	size := rows * cols
	return &Dense{
		rows:   rows,
		cols:   cols,
		size:   size,
		data:   make([]float64, size),
		viewOf: nil,
	}
}

// NewEmptyVecDense returns a new one-hot vector of the given size.
func OneHotVecDense(size int, oneAt int) *Dense {
	if oneAt >= size {
		panic(fmt.Sprintf("mat: impossible to set the one at index %d. The size is: %d", oneAt, size))
	}
	vec := NewEmptyVecDense(size)
	vec.SetVec(oneAt, 1.0)
	return vec
}

// NewInitDense returns a new rows x cols dense matrix initialized with a constant value.
func NewInitDense(rows, cols int, val float64) *Dense {
	out := NewEmptyDense(rows, cols)
	f64.AddConst(val, out.data)
	return out
}

// NewInitDense returns a new size x 1 dense matrix initialized with a constant value.
func NewInitVecDense(size int, val float64) *Dense {
	return NewInitDense(size, 1, val)
}

// SetData sets the data
func (d *Dense) SetData(data []float64) {
	if len(data) != d.size {
		panic(fmt.Sprintf("mat: data size must be: %d", d.size))
	}
	_ = append(d.data[:0], data...)
}

// ZerosLike returns a new Dense with of the same dimensions of the receiver, initialized with zeros.
func (d *Dense) ZerosLike() Matrix {
	return NewEmptyDense(d.rows, d.cols)
}

// ZerosLike returns a new Dense with of the same dimensions of the receiver, initialized to ones.
func (d *Dense) OnesLike() Matrix {
	buf := d.ZerosLike()
	f64.AddConst(1.0, buf.(*Dense).data)
	return buf
}

// Clone returns a new matrix copying the values of the receiver.
func (d *Dense) Clone() Matrix {
	return NewDense(d.rows, d.cols, d.data)
}

// Copy copies the data to the receiver.
func (d *Dense) Copy(other Matrix) {
	if !SameDims(d, other) {
		panic("mat: incompatible matrix dimensions.")
	}
	if other, ok := other.(*Dense); !ok {
		panic("mat: incompatible matrix types.")
	} else {
		_ = append(d.data[:0], other.data...)
	}
}

// View returns a new Matrix sharing the same underlying data.
func (d *Dense) View(rows, cols int) *Dense {
	if d.Size() != rows*cols {
		panic("mat: incompatible sizes.")
	}
	return &Dense{
		rows:   rows,
		cols:   cols,
		size:   rows * cols,
		data:   d.data,
		viewOf: d,
	}
}

// Zeros set all the values to zeros.
func (d *Dense) Zeros() {
	data := d.data // avoid bounds check
	for i := range data {
		data[i] = 0.0
	}
}

// Dims returns the number of rows and columns.
func (d *Dense) Dims() (r, c int) {
	return d.rows, d.cols
}

// Rows returns the number of rows.
func (d *Dense) Rows() int {
	return d.rows
}

// Columns returns the number of columns.
func (d *Dense) Columns() int {
	return d.cols
}

// Size returns the size of the matrix (rows * cols).
func (d *Dense) Size() int {
	return d.size
}

// LastIndex returns the last index.
func (d *Dense) LastIndex() int {
	return d.size - 1
}

// Data returns the underlying data.
func (d *Dense) Data() []float64 {
	return d.data
}

// IsVectors returns whether the matrix has one row or one column, or not.
func (d *Dense) IsVector() bool {
	return d.rows == 1 || d.cols == 1
}

// IsScalar returns whether the matrix contains a scalar, or not.
func (d *Dense) IsScalar() bool {
	return d.size == 1
}

// Scalar returns the scalar. It panics if the matrix contains more elements.
func (d *Dense) Scalar() float64 {
	if !d.IsScalar() {
		panic("mat: expected scalar but the matrix contains more elements.")
	}
	return d.data[0]
}

// Set sets the value v at row i and column j.
func (d *Dense) Set(i int, j int, v float64) {
	if i >= d.rows {
		panic("mat: 'i' argument out of range.")
	}
	if j >= d.cols {
		panic("mat: 'j' argument out of range")
	}
	d.data[i*d.cols+j] = v
}

// At returns the value at row i and column j.
func (d *Dense) At(i int, j int) float64 {
	if i >= d.rows {
		panic("mat: 'i' argument out of range.")
	}
	if j >= d.cols {
		panic("mat: 'j' argument out of range")
	}
	return d.data[i*d.cols+j]
}

// SetVec sets the value v at position i of a vector.
// It panics if not IsVector().
func (d *Dense) SetVec(i int, v float64) {
	if !(d.IsVector()) {
		panic("mat: expected vector")
	}
	if i >= d.rows {
		panic("mat: 'i' argument out of range.")
	}
	d.data[i] = v
}

// AtVec returns the value at position i of a vector
// It panics if not IsVector().
func (d *Dense) AtVec(i int) float64 {
	if !(d.IsVector()) {
		panic("mat: expected vector")
	}
	if i >= d.rows {
		panic("mat: 'i' argument out of range.")
	}
	return d.data[i]
}

// ExtractRow returns a copy of the i-th row of the matrix.
func (d *Dense) ExtractRow(i int) Matrix {
	if i >= d.Rows() {
		panic("mat: index out of range")
	}
	out := NewVecDense(d.data[i*d.cols : i*d.cols+d.cols])
	return out
}

// ExtractRow returns a copy of the i-th column of the matrix.
func (d *Dense) ExtractColumn(i int) Matrix {
	if i >= d.Columns() {
		panic("mat: index out of range")
	}
	out := NewEmptyVecDense(d.rows)
	data := out.data
	for k := range data {
		data[k] = d.data[k*d.cols+i]
	}
	return out
}

// T returns the transpose of the matrix.
func (d *Dense) T() Matrix {
	r, c := d.Dims()
	m := NewEmptyDense(c, r)
	length := len(m.data)
	index := 0
	for _, value := range d.data {
		m.data[index] = value
		index += r
		if index >= length {
			index -= length - 1
		}
	}
	return m
}

// Reshape returns a copy of the matrix. It panics if the dimensions are not compatible.
func (d *Dense) Reshape(r, c int) Matrix {
	if d.Size() != r*c {
		panic("mat: incompatible sizes.")
	}
	return NewDense(r, c, d.data)
}

// ApplyWithAlpha executes the unary function fn, taking additional parameters alpha.
func (d *Dense) ApplyWithAlpha(fn func(i, j int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64) {
	if !SameDims(d, a) {
		panic("mat: incompatible matrix dimensions.")
	}
	for i := 0; i < d.rows; i++ {
		for j := 0; j < d.cols; j++ {
			d.data[i*d.cols+j] = fn(i, j, a.At(i, j), alpha...)
		}
	}
}

// Apply execute the unary function fn.
func (d *Dense) Apply(fn func(i, j int, v float64) float64, a Matrix) {
	if !SameDims(d, a) {
		panic("mat: incompatible matrix dimensions.")
	}
	for i := 0; i < d.rows; i++ {
		for j := 0; j < d.cols; j++ {
			d.data[i*d.cols+j] = fn(i, j, a.At(i, j))
		}
	}
}

// AddScalar performs an addition between the Matrix and a float.
func (d *Dense) AddScalar(n float64) Matrix {
	out := d.Clone().(*Dense)
	f64.AddConst(n, out.data)
	return out
}

// SubScalar performs a subtraction between the Matrix and a float.
func (d *Dense) SubScalar(n float64) Matrix {
	out := d.Clone().(*Dense)
	f64.AddConst(-n, out.data)
	return out
}

// AddScalarInPlace adds the scalar to the receiver.
func (d *Dense) AddScalarInPlace(n float64) Matrix {
	f64.AddConst(n, d.data)
	return d
}

// SubScalarInPlace subtracts the scalar to the receiver.
func (d *Dense) SubScalarInPlace(n float64) Matrix {
	for i := 0; i < len(d.data); i++ {
		d.data[i] -= n
	}
	return d
}

// ProdScalarInPlace multiply a float with the receiver in place.
func (d *Dense) ProdScalarInPlace(n float64) Matrix {
	f64.ScalUnitary(n, d.data)
	return d
}

// ProdMatrixScalarInPlace multiply a matrix with a float, storing the result in the receiver.
func (d *Dense) ProdMatrixScalarInPlace(m Matrix, n float64) Matrix {
	f64.ScalUnitaryTo(d.data, n, m.(*Dense).data)
	return d
}

// ProdScalar returns the multiplication of the float with the receiver.
func (d *Dense) ProdScalar(n float64) Matrix {
	out := d.ZerosLike().(*Dense)
	f64.ScalUnitaryTo(out.data, n, d.data)
	return out
}

// Add returns the addition with a matrix with the receiver.
func (d *Dense) Add(other Matrix) Matrix {
	if !(SameDims(d, other) ||
		(other.Columns() == 1 && other.Rows() == d.Rows()) ||
		(other.IsVector() && d.IsVector() && other.Size() == d.Size())) {
		panic("mat: matrices with not compatible size")
	}
	b := other.(*Dense)
	out := d.ZerosLike().(*Dense)
	f64.AxpyUnitaryTo(out.data, 1.0, b.data, d.data)
	return out
}

// AddInPlace performs the addition with the other matrix in place.
func (d *Dense) AddInPlace(other Matrix) Matrix {
	if !(SameDims(d, other) ||
		(other.Columns() == 1 && other.Rows() == d.Rows()) ||
		(other.IsVector() && d.IsVector() && other.Size() == d.Size())) {
		panic("mat: matrices with not compatible size")
	}
	b := other.(*Dense)
	f64.AxpyUnitary(1.0, b.data, d.data)
	return d
}

// Sub returns the subtraction with a matrix with the receiver.
func (d *Dense) Sub(other Matrix) Matrix {
	if !(SameDims(d, other) ||
		(other.Columns() == 1 && other.Rows() == d.Rows()) ||
		(other.IsVector() && d.IsVector() && other.Size() == d.Size())) {
		panic("mat: matrices with not compatible size")
	}
	out := d.ZerosLike().(*Dense)
	b := other.(*Dense)
	f64.AxpyUnitaryTo(out.data, -1.0, b.data, d.data)
	return out
}

// SubInPlace performs the subtraction with the other matrix in place.
func (d *Dense) SubInPlace(other Matrix) Matrix {
	if !(SameDims(d, other) ||
		(other.Columns() == 1 && other.Rows() == d.Rows()) ||
		(other.IsVector() && d.IsVector() && other.Size() == d.Size())) {
		panic("mat: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense:
		f64.AxpyUnitary(-1.0, other.data, d.data)
	case *Sparse:
		other.DoNonZero(func(i, j int, k float64) {
			d.Set(i, j, d.At(i, j)-k)
		})
	}
	return d
}

// Prod performs the element-wise product with the receiver.
func (d *Dense) Prod(other Matrix) Matrix {
	if !(SameDims(d, other) ||
		(other.Columns() == 1 && other.Rows() == d.Rows()) ||
		(other.IsVector() && d.IsVector() && other.Size() == d.Size())) {
		panic("mat: matrices with not compatible size")
	}
	out := d.ZerosLike().(*Dense)
	b := other.(*Dense)
	for i, val := range d.data {
		out.data[i] = val * b.data[i]
	}
	return out
}

// ProdInPlace performs the element-wise product with the receiver in place.
func (d *Dense) ProdInPlace(other Matrix) Matrix {
	if !(SameDims(d, other) ||
		(other.Columns() == 1 && other.Rows() == d.Rows()) ||
		(other.IsVector() && d.IsVector() && other.Size() == d.Size())) {
		panic("mat: matrices with not compatible size")
	}
	b := other.(*Dense)
	for i, val := range b.data {
		d.data[i] *= val
	}
	return d
}

// Div returns the result of the element-wise division.
func (d *Dense) Div(other Matrix) Matrix {
	if !(SameDims(d, other) ||
		(other.Columns() == 1 && other.Rows() == d.Rows()) ||
		(other.IsVector() && d.IsVector() && other.Size() == d.Size())) {
		panic("mat: matrices with not compatible size")
	}
	out := d.ZerosLike().(*Dense)
	f64.DivTo(out.data, d.data, other.(*Dense).data)
	return out
}

// Div performs the result of the element-wise division in place.
func (d *Dense) DivInPlace(other Matrix) Matrix {
	if !(SameDims(d, other) ||
		(other.Columns() == 1 && other.Rows() == d.Rows()) ||
		(other.IsVector() && d.IsVector() && other.Size() == d.Size())) {
		panic("mat: matrices with not compatible size")
	}
	b := other.(*Dense)
	for i, val := range b.data {
		d.data[i] *= 1.0 / val
	}
	return d
}

// Mul performs the multiplication row by column. AB = C
// if A is an r x c Matrix, and B is j X k, c = j the resulting Matrix C will be r x k
func (d *Dense) Mul(other Matrix) Matrix {
	if d.Columns() != other.Rows() {
		panic("mat: matrices with not compatible size")
	}
	out := NewEmptyDense(d.Rows(), other.Columns())

	switch b := other.(type) {
	case *Dense:
		if out.cols == 1 {
			f64.GemvN(
				uintptr(d.rows), // m
				uintptr(d.cols), // n
				1.0,             // alpha
				d.data,          // a
				uintptr(d.cols), // lda
				b.data,          // x
				1.0,             // incX
				0.0,             // beta
				out.data,        // y
				1.0,             // incY
			)
		} else {
			f64.DgemmSerial(
				false,
				false,
				d.rows,   // m
				b.cols,   // n
				d.cols,   // k
				d.data,   // a
				d.cols,   // lda
				b.data,   // b
				b.cols,   // ldb
				out.data, // c
				out.cols, // ldc
				1.0,      // alpha
			)

			/*
				// parallel implementation
				f64.Dgemm(
					false,    // aTrans
					false,    // bTrans
					d.rows,   // m
					b.cols,   // n
					d.cols,   // k
					1.0,      // alpha
					d.data,   // a
					d.cols,   // lda
					b.data,   // b
					b.cols,   // ldb
					0.0,      // beta
					out.data, // c
					out.cols, // ldc
				)

			*/

		}

		return out

	case *Sparse:
		b.DoNonZero(func(k, j int, v float64) {
			for i := 0; i < d.Rows(); i++ {
				out.Set(i, j, out.At(i, j)+d.At(i, k)*v)
			}
		})
	}
	return out
}

// MulT performs the matrix multiplication row by column. ATB = C, where AT is the transpose of B
// if A is an r x c Matrix, and B is j x k, r = j the resulting Matrix C will be c x k
func (d *Dense) MulT(other Matrix) Matrix {
	if d.Rows() != other.Rows() {
		panic("mat: matrices with not compatible size")
	}
	out := NewEmptyDense(d.Columns(), other.Columns())

	switch b := other.(type) {
	case *Dense:
		if out.cols == 1 {
			f64.GemvT(
				uintptr(d.rows), // m
				uintptr(d.cols), // n
				1.0,             // alpha
				d.data,          // a
				uintptr(d.cols), // lda
				b.data,          // x
				1.0,             // incX
				0.0,             // beta
				out.data,        // y
				1.0,             // incY
			)
		} else {
			panic("mat: matrices with not compatible size")
		}
		return out
	case *Sparse:
		panic("mat: matrices not compatible")
	}
	return out
}

// DotUnitary returns the dot product of two vectors.
func (d *Dense) DotUnitary(other Matrix) float64 {
	if d.Size() != other.Size() {
		panic("mat: incompatible sizes.")
	}
	return f64.DotUnitary(d.data, other.Data())
}

// ClipInPlace performs the clip in place.
// If element k of Matrix if k > max, k = max and if k < min, k = min
func (d *Dense) ClipInPlace(min, max float64) Matrix {
	d.Apply(func(i, j int, v float64) float64 {
		if v < min {
			return min
		} else if v > max {
			return max
		} else {
			return v
		}
	}, d)
	return d
}

// Abs returns a new matrix applying the abs function to all elements.
func (d *Dense) Abs() Matrix {
	out := d.ZerosLike()
	out.Apply(func(i, j int, v float64) float64 {
		return math.Abs(v)
	}, d)
	return out
}

// Pow returns a new matrix applying the power v (applying the pow function) to all elements.
func (d *Dense) Pow(power float64) Matrix {
	out := d.Clone().(*Dense)
	out.Apply(func(i, j int, v float64) float64 {
		return math.Pow(v, power)
	}, d)
	return out
}

// Sqrt returns a new matrix applying the sqrt function to all elements.
func (d *Dense) Sqrt() Matrix {
	out := d.ZerosLike().(*Dense)
	for i, val := range d.data {
		out.data[i] = math.Sqrt(val)
	}
	return out
}

// Sum returns the sum of all values of the matrix.
func (d *Dense) Sum() float64 {
	sum := 0.0
	for _, elem := range d.data {
		sum += elem
	}
	return sum
}

// Max returns the max value of the matrix.
func (d *Dense) Max() float64 {
	r, c := d.Dims()
	max := math.Inf(-1)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := d.At(i, j)
			if v > max {
				max = v
			}
		}
	}
	return max
}

// Min returns the min value of the matrix.
func (d *Dense) Min() float64 {
	r, c := d.Dims()
	min := math.Inf(1)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := d.At(i, j)
			if v < min {
				min = v
			}
		}
	}
	return min
}

// Range extracts data from the the Matrix from elements start (inclusive) and end (exclusive).
func (d *Dense) Range(start, end int) Matrix {
	data := make([]float64, end-start)
	for k := 0; k < end-start; k++ {
		data[k] = d.data[start+k]
	}
	return NewVecDense(data)
}

// SplitV extract N vectors from the matrix d.
// N[i] has size sizes[i].
func (d *Dense) SplitV(sizes ...int) []Matrix {
	out := make([]Matrix, len(sizes))
	offset := 0
	for i := 0; i < len(sizes); i++ {
		startIndex := offset
		offset = startIndex + sizes[i]
		out[i] = d.Range(startIndex, offset)
	}
	return out
}

// Norm returns the vector norm.  Use pow = 2.0 for Euclidean.
func (d *Dense) Norm(pow float64) float64 {
	s := 0.0
	for _, x := range d.data {
		s += math.Pow(x, pow)
	}
	return math.Pow(s, 1/pow)
}

// Maximum returns a new matrix containing the element-wise maxima.
func (d *Dense) Maximum(other Matrix) *Dense {
	if d.Columns() != other.Columns() && d.Rows() != other.Rows() {
		panic("mat: matrix with not compatible size")
	}
	out := NewEmptyDense(d.rows, d.cols)
	for i := 0; i < d.rows; i++ {
		for j := 0; j < d.cols; j++ {
			a := d.At(i, j)
			b := other.At(i, j)
			if a > b {
				out.data[i*d.cols+j] = a
			} else {
				out.data[i*d.cols+j] = b
			}
		}
	}
	return out
}

// Augment places the identity matrix at the end of the original matrix
func (d *Dense) Augment() Matrix {
	if d.Columns() != d.Rows() {
		panic("mat: matrix must be square")
	}
	out := NewEmptyDense(d.rows, d.rows+d.cols)
	for i := 0; i < d.rows; i++ {
		for j := 0; j < d.cols; j++ {
			out.Set(i, j, d.At(i, j))
		}
		out.Set(i, i+d.rows, 1.0)
	}
	return out
}

// SwapInPlace swaps two rows of the matrix in place
func (d Dense) SwapInPlace(r1, r2 int) {
	if d.IsVector() {
		panic("mat: input must be a matrix")
	}
	if r1 >= d.rows || r2 >= d.rows {
		panic("mat: index out of range")
	}
	temp := NewVecDense(d.ExtractRow(r1).Data())
	for j := 0; j < d.cols; j++ {
		d.data[r1*d.cols+j] = d.data[r2*d.cols+j]
	}
	for j := 0; j < d.cols; j++ {
		d.data[r2*d.cols+j] = temp.data[j]
	}
}

// Return the partial pivots of a square matrix to reorder rows.
// Considerate square sub-matrix from element (offset, offset).
func (d *Dense) Pivoting(row int) (Matrix, bool, []int) {
	if d.Columns() != d.Rows() {
		panic("mat: matrix must be square")
	}
	pv := make([]int, d.cols)
	positions := make([]int, 2)
	for i := range pv {
		pv[i] = i
	}
	j := row
	max := math.Abs(d.data[row*d.cols+j])
	for i := row; i < d.cols; i++ {
		if d.data[i*d.cols+j] > max {
			max = math.Abs(d.data[i*d.cols+j])
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

	p := NewEmptyDense(d.cols, d.cols)
	for r, c := range pv {
		p.data[r*d.cols+c] = 1
	}
	return p, swap, positions
}

// I a.k.a identity returns square matrix with ones on the diagonal and zeros elsewhere.
func I(size int) *Dense {
	out := NewEmptyDense(size, size)
	for i := 0; i < size; i++ {
		out.Set(i, i, 1.0)
	}
	return out
}

// LU performs lower–upper (LU) decomposition of a square matrix D such as PLU = D, L is lower diagonal and U is upper diagonal, p are pivots.
func (d *Dense) LU() (l, u, p *Dense) {
	if d.Columns() != d.Rows() {
		panic("mat: matrix must be square")
	}
	u = d.Clone().(*Dense)
	p = I(d.cols)
	l = NewEmptyDense(d.cols, d.cols)
	for i := 0; i < d.cols; i++ {
		_, swap, positions := u.Pivoting(i)
		if swap {
			u.SwapInPlace(positions[0], positions[1])
			p.SwapInPlace(positions[0], positions[1])
			l.SwapInPlace(positions[0], positions[1])
		}
		lt := I(d.cols)
		for k := i + 1; k < d.cols; k++ {
			lt.Data()[k*d.cols+i] = -u.Data()[k*d.cols+i] / (u.Data()[i*d.cols+i])
			l.Data()[k*d.cols+i] = u.Data()[k*d.cols+i] / (u.Data()[i*d.cols+i])
		}
		u = lt.Mul(u).(*Dense)
	}
	for i := 0; i < d.cols; i++ {
		l.Data()[i*d.cols+i] = 1.0
	}
	return
}

// Inverse returns the inverse of the matrix.
func (d Dense) Inverse() Matrix {
	if d.Columns() != d.Rows() {
		panic("mat: matrix must be square")
	}
	out := NewEmptyDense(d.cols, d.cols)
	s := NewEmptyDense(d.cols, d.cols)
	l, u, p := d.LU()
	for b := 0; b < d.cols; b++ {
		// find solution of Ly = b
		for i := 0; i < l.Rows(); i++ {
			sum := 0.0
			for j := 0; j < i; j++ {
				sum += l.Data()[i*d.cols+j] * s.data[j*d.cols+b]
			}
			s.data[i*d.cols+b] = p.Data()[i*d.cols+b] - sum
		}
		// find solution of Ux = y
		for i := d.cols - 1; i >= 0; i-- {
			sum := 0.0
			for j := i + 1; j < d.cols; j++ {
				sum += u.Data()[i*d.cols+j] * out.data[j*d.cols+b]
			}
			out.data[i*d.cols+b] = (1.0 / u.Data()[i*d.cols+i]) * (s.data[i*d.cols+b] - sum)
		}
	}
	return out
}

// String returns the string representation of the data.
func (d *Dense) String() string {
	return fmt.Sprintf("%v", d.data)
}
