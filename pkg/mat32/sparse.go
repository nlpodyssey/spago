// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"encoding/gob"
	"fmt"
	"math"
)

var _ Matrix = &Sparse{}

// Sparse is the implementation of a sparse matrix that uses Float as data type.
type Sparse struct {
	rows       int
	cols       int
	size       int     // rows*cols
	nzElements []Float // A vector
	nnzRow     []int   // IA vector
	colsIndex  []int   // JA vector
}

func init() {
	gob.Register(&Sparse{})
}

// NewSparse returns a new rows x cols sparse matrix populated with a copy of the non-zero elements.
// The elements cannot be nil, panic otherwise. Use NewEmptySparse to initialize an empty matrix.
func NewSparse(rows, cols int, elements []Float) *Sparse {
	if elements == nil {
		panic("mat32: elements cannot be nil. Use NewEmptySparse() instead.")
	}
	if len(elements) != rows*cols {
		panic(fmt.Sprintf("mat32: wrong matrix dimensions. Elements size must be: %d", rows*cols))
	}
	return newSparse(rows, cols, elements)
}

// NewVecSparse returns a new column sparse vector populated with the non-zero elements.
// The elements cannot be nil, panic otherwise. Use NewEmptyVecSparse to initialize an empty matrix.
func NewVecSparse(elements []Float) *Sparse {
	if elements == nil {
		panic("mat32: elements cannot be nil. Use NewEmptyVecSparse() instead.")
	}
	return newSparse(len(elements), 1, elements)
}

// NewEmptyVecSparse returns a new sparse vector of the given size.
func NewEmptyVecSparse(size int) *Sparse {
	return NewEmptySparse(size, 1)
}

// NewEmptySparse returns a new rows x cols Sparse matrix.
func NewEmptySparse(rows, cols int) *Sparse {
	return newSparse(rows, cols, make([]Float, rows*cols))
}

func newSparse(rows, cols int, elements []Float) *Sparse {
	nzElements := make([]Float, 0)
	nnzRow := make([]int, rows+1)
	colsIndex := make([]int, 0)
	k := 0
	nnz := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if elements[k] != 0.0 {
				nzElements = append(nzElements, elements[k])
				colsIndex = append(colsIndex, j)
				nnz++
			}
			k++
		}
		nnzRow[i+1] = nnz
	}
	return &Sparse{
		rows:       rows,
		cols:       cols,
		size:       rows * cols,
		nzElements: nzElements,
		nnzRow:     nnzRow,
		colsIndex:  colsIndex,
	}
}

// Coordinate represents the row I and column J of a Sparse matrix.
type Coordinate struct {
	I, J int
}

// NewSparseFromMap creates a new Sparse matrix from a raw map of values.
func NewSparseFromMap(rows, cols int, elements map[Coordinate]Float) *Sparse {
	nzElements := make([]Float, 0)
	nnzRow := make([]int, rows+1)
	colsIndex := make([]int, 0)
	nnz := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			k := Coordinate{I: i, J: j}
			if element, ok := elements[k]; ok && element != 0.0 {
				nzElements = append(nzElements, element)
				colsIndex = append(colsIndex, j)
				nnz++
			}
		}
		nnzRow[i+1] = nnz
	}
	return &Sparse{
		rows:       rows,
		cols:       cols,
		size:       rows * cols,
		nzElements: nzElements,
		nnzRow:     nnzRow,
		colsIndex:  colsIndex,
	}
}

// OneHotSparse creates a new one-hot Sparse vector. It panics if oneAt is an
// invalid index.
func OneHotSparse(size int, oneAt int) *Sparse {
	if oneAt >= size {
		panic(fmt.Sprintf("mat32: impossible to set the one at index %d. The size is: %d", oneAt, size))
	}
	vec := NewEmptyVecSparse(size)
	vec.nzElements = append(vec.nzElements, 1.0)
	vec.colsIndex = append(vec.colsIndex, 0)
	for i := oneAt + 1; i < size; i++ {
		vec.nnzRow[i] = 1
	}
	return vec
}

// Sparsity returns the sparsity of the Sparse matrix.
func (s *Sparse) Sparsity() Float {
	return Float(s.size-len(s.nzElements)) / Float(s.size)
}

// ToDense transforms a Sparse matrix into a new Dense matrix.
func (s *Sparse) ToDense() *Dense {
	out := NewEmptyDense(s.rows, s.cols)
	for i := 0; i < s.rows; i++ {
		for elem := s.nnzRow[i]; elem < s.nnzRow[i+1]; elem++ {
			out.data[i*s.cols+s.colsIndex[elem]] = s.nzElements[elem]
		}
	}
	return out
}

// ZerosLike returns a new Sparse matrix with the same dimensions of the receiver,
// initialized with zeroes.
func (s *Sparse) ZerosLike() Matrix {
	return NewEmptySparse(s.Dims())
}

// OnesLike is currently not implemented for a Sparse matrix (it always panics).
func (s *Sparse) OnesLike() Matrix {
	panic("mat32: OnesLike not implemented for Sparse matrices")
}

// Clone returns a new Sparse matrix, copying all its values from the receiver.
func (s *Sparse) Clone() Matrix {
	return NewSparse(s.rows, s.cols, s.Data())
}

// Copy copies the data from the other matrix to the receiver.
// It panics if the matrices have different dimensions, or if the other
// matrix is not Sparse.
func (s *Sparse) Copy(other Matrix) {
	if !SameDims(s, other) {
		panic("mat32: incompatible matrix dimensions.")
	}
	if other, ok := other.(*Sparse); !ok {
		panic("mat32: incompatible matrix types.")
	} else {
		s.colsIndex = append(s.colsIndex[:0], other.colsIndex...)
		s.nnzRow = append(s.nnzRow[:0], other.nnzRow...)
		s.nzElements = append(s.nzElements[:0], other.nzElements...)
	}
}

// Zeros sets all the values of the matrix to zero.
func (s *Sparse) Zeros() {
	s.nzElements = make([]Float, 0)
	s.nnzRow = make([]int, s.rows+1)
	s.nnzRow[0] = 0
	s.colsIndex = make([]int, 0)
}

// Dims returns the number of rows and columns of the matrix.
func (s *Sparse) Dims() (r, c int) {
	return s.rows, s.cols
}

// Rows returns the number of rows of the matrix.
func (s *Sparse) Rows() int {
	return s.rows
}

// Columns returns the number of columns of the matrix.
func (s *Sparse) Columns() int {
	return s.cols
}

// Size returns the size of the matrix (rows × columns).
func (s *Sparse) Size() int {
	return s.size
}

// LastIndex returns the last element's index, in respect of linear indexing.
// It returns -1 if the matrix is empty.
func (s *Sparse) LastIndex() int {
	return s.size - 1
}

// Data returns the underlying data of the matrix, as a raw one-dimensional slice of values.
func (s *Sparse) Data() []Float {
	out := make([]Float, s.rows*s.cols)
	for i := 0; i < s.rows; i++ {
		for elem := s.nnzRow[i]; elem < s.nnzRow[i+1]; elem++ {
			out[i*s.cols+s.colsIndex[elem]] = s.nzElements[elem]
		}
	}
	return out
}

// IsVector returns whether the matrix is either a row or column vector.
func (s *Sparse) IsVector() bool {
	return s.rows == 1 || s.cols == 1
}

// IsScalar returns whether the matrix contains exactly one scalar value.
func (s *Sparse) IsScalar() bool {
	return s.size == 1
}

// Scalar returns the scalar value.
// It panics if the matrix does not contain exactly one element.
func (s *Sparse) Scalar() Float {
	if !s.IsScalar() {
		panic("mat32: expected scalar but the matrix contains more elements.")
	}
	if len(s.nzElements) > 0 {
		return s.nzElements[0]
	}
	return 0.0
}

// Set sets the value v at row i and column j.
// It panics if the given indices are out of range.
func (s *Sparse) Set(i int, j int, v Float) {
	panic("mat32: Set not implemented for Sparse matrices")
}

// At returns the value at row i and column j.
// It panics if the given indices are out of range.
func (s *Sparse) At(i int, j int) Float {
	if i >= s.rows {
		panic("mat32: 'i' argument out of range.")
	}
	if j >= s.cols {
		panic("mat32: 'j' argument out of range")
	}
	for k := s.nnzRow[i]; k < s.nnzRow[i+1]; k++ {
		if j == s.colsIndex[k] {
			return s.nzElements[k]
		}
	}
	return 0.0
}

// SetVec is currently not implemented for a Sparse matrix (it always panics).
func (s *Sparse) SetVec(i int, v Float) {
	panic("mat32: SetVec not implemented for Sparse matrices")
}

// AtVec returns the value at position i of a vector.
// It panics if the receiver is not a vector.
func (s *Sparse) AtVec(i int) Float {
	if !(s.IsVector()) {
		panic("mat32: expected vector")
	}
	if i >= s.size {
		panic("mat32: 'i' argument out of range.")
	}
	if s.cols == 1 {
		if (s.nnzRow[i+1] - s.nnzRow[i]) > 0 {
			return s.nzElements[s.nnzRow[i]]
		}
	}
	if s.rows == 1 {
		for k, j := range s.colsIndex {
			if i == j {
				return s.nzElements[k]
			}
		}
	}
	return 0.0
}

// DoNonZero calls a function for each non-zero element of the matrix.
// The parameters of the function are the element indices and its value.
func (s *Sparse) DoNonZero(fn func(i, j int, v Float)) {
	for i := 0; i < s.rows; i++ {
		for elem := s.nnzRow[i]; elem < s.nnzRow[i+1]; elem++ {
			j := s.colsIndex[elem]
			v := s.nzElements[elem]
			fn(i, j, v)
		}
	}
}

// T returns the transpose of the matrix.
func (s *Sparse) T() Matrix {
	// Convert CSR to CSC
	out := NewEmptySparse(s.cols, s.rows)
	out.nzElements = make([]Float, len(s.nzElements))
	out.colsIndex = make([]int, len(s.nzElements))

	for _, c := range s.colsIndex {
		out.nnzRow[c]++
	}
	nnz := 0
	for i := 0; i < s.cols; i++ {
		var temp = out.nnzRow[i]
		out.nnzRow[i] = nnz
		nnz += temp
	}
	out.nnzRow[s.cols] = len(s.nzElements)
	s.DoNonZero(func(i, j int, v Float) {
		var destination = out.nnzRow[j]
		out.nzElements[destination] = v
		out.colsIndex[destination] = i
		out.nnzRow[j]++
	})
	last := 0
	for i := 0; i < s.cols; i++ {
		var temp = out.nnzRow[i]
		out.nnzRow[i] = last
		last = temp
	}
	return out
}

// Reshape is currently not implemented for a Sparse matrix (it always panics).
func (s *Sparse) Reshape(r, c int) Matrix {
	panic("mat32: Reshape not implemented for Sparse matrices")
}

// Apply executes the unary function fn.
// Important: apply to Functions such that f(0) = 0 (i.e. Sin, Tan)
func (s *Sparse) Apply(fn func(i, j int, v Float) Float, a Matrix) {
	if _, ok := a.(*Sparse); !ok {
		panic("mat32: incompatible matrix types.")
	}
	for i := 0; i < len(s.nzElements); i++ {
		s.nzElements[i] = fn(i, 0, a.(*Sparse).nzElements[i])
	}
}

// ApplyWithAlpha is currently not implemented for a Sparse matrix (it always panics).
func (s *Sparse) ApplyWithAlpha(fn func(i, j int, v Float, alpha ...Float) Float, a Matrix, alpha ...Float) {
	panic("mat32: ApplyWithAlpha not implemented for Sparse matrices")
}

// AddScalar performs the addition between the matrix and the given value,
// returning a new Dense matrix.
func (s *Sparse) AddScalar(n Float) Matrix {
	out := NewInitDense(s.rows, s.cols, n)
	s.DoNonZero(func(i, j int, v Float) {
		out.Data()[i*s.cols+j] += v
	})

	return out
}

// AddScalarInPlace is currently not implemented for a Sparse matrix (it always panics).
func (s *Sparse) AddScalarInPlace(n Float) Matrix {
	panic("mat32: AddScalarInPlace not implemented for Sparse matrices")
}

// SubScalar performs a subtraction between the matrix and the given value,
// returning a new Dense matrix.
func (s *Sparse) SubScalar(n Float) Matrix {
	out := NewInitDense(s.rows, s.cols, -n)
	s.DoNonZero(func(i, j int, v Float) {
		out.Data()[i*s.cols+j] += v
	})
	return out
}

// SubScalarInPlace is currently not implemented for a Sparse matrix (it always panics).
func (s *Sparse) SubScalarInPlace(n Float) Matrix {
	panic("mat32: SubScalarInPlace not implemented for Sparse matrices")
}

// ProdScalar returns the multiplication between the matrix and the given value,
// returning a new Sparse matrix.
func (s *Sparse) ProdScalar(n Float) Matrix {
	out := s.Clone().(*Sparse) // TODO: find a better alternative to s.Clone()
	if n == 0.0 {
		return NewEmptySparse(s.rows, s.cols)
	}
	for i, elem := range s.nzElements {
		out.nzElements[i] = elem * n
	}
	return out
}

// ProdScalarInPlace performs the in-place multiplication between the matrix and
// the given value, returning the same receiver Sparse matrix.
func (s *Sparse) ProdScalarInPlace(n Float) Matrix {
	if n == 0.0 {
		*s = *NewEmptySparse(s.rows, s.cols)
		return s
	}
	for i, elem := range s.nzElements {
		s.nzElements[i] = elem * n
	}
	return s
}

// ProdMatrixScalarInPlace multiplies the given matrix with the value, storing the
// result in the receiver, and returning the same receiver Sparse matrix.
func (s *Sparse) ProdMatrixScalarInPlace(m Matrix, n Float) Matrix {
	if _, ok := m.(*Sparse); !ok {
		panic("mat32: incompatible matrix types.")
	}
	if !SameDims(s, m) {
		panic("mat32: incompatible matrix dimensions.")
	}
	if n == 0.0 {
		*s = *NewEmptySparse(s.rows, s.cols)
		return s
	}
	for _, elem := range m.(*Sparse).colsIndex {
		s.colsIndex = append(s.colsIndex, elem)
	}
	for i := 0; i < len(m.(*Sparse).nnzRow); i++ {
		s.nnzRow[i] = m.(*Sparse).nnzRow[i]
	}
	for _, elem := range m.(*Sparse).nzElements {
		s.nzElements = append(s.nzElements, elem*n)
	}
	return s
}

func (s *Sparse) addSparse(other *Sparse) *Sparse {
	out := NewEmptySparse(s.rows, s.cols)
	var nnzElements = 0
	for i := 0; i < s.rows; i++ {
		var sPos, otherPos = s.nnzRow[i], other.nnzRow[i]
		out.nnzRow[0] = 0
		for sPos < s.nnzRow[i+1] && otherPos < other.nnzRow[i+1] {
			if s.colsIndex[sPos] < other.colsIndex[otherPos] {
				out.colsIndex = append(out.colsIndex, s.colsIndex[sPos])
				out.nzElements = append(out.nzElements, s.nzElements[sPos])
				sPos++
				nnzElements++
			} else if s.colsIndex[sPos] > other.colsIndex[otherPos] {
				out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
				out.nzElements = append(out.nzElements, other.nzElements[otherPos])
				otherPos++
				nnzElements++
			} else if s.colsIndex[sPos] == other.colsIndex[otherPos] {
				var result = other.nzElements[otherPos] + s.nzElements[sPos]
				if result != 0.0 {
					out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
					out.nzElements = append(out.nzElements, result)
					nnzElements++
				}
				sPos++
				otherPos++
			}
		}

		for sPos < s.nnzRow[i+1] {
			out.colsIndex = append(out.colsIndex, s.colsIndex[sPos])
			out.nzElements = append(out.nzElements, s.nzElements[sPos])
			sPos++
			nnzElements++
		}
		for otherPos < other.nnzRow[i+1] {
			out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
			out.nzElements = append(out.nzElements, other.nzElements[otherPos])
			otherPos++
			nnzElements++

		}
		out.nnzRow[i+1] = nnzElements
	}
	return out
}

func (s *Sparse) subSparse(other *Sparse) *Sparse {
	out := NewEmptySparse(s.rows, s.cols)
	var nnzElements = 0
	for i := 0; i < s.rows; i++ {
		var sPos, otherPos = s.nnzRow[i], other.nnzRow[i]
		out.nnzRow[0] = 0
		for sPos < s.nnzRow[i+1] && otherPos < other.nnzRow[i+1] {
			if s.colsIndex[sPos] < other.colsIndex[otherPos] {
				out.colsIndex = append(out.colsIndex, s.colsIndex[sPos])
				out.nzElements = append(out.nzElements, s.nzElements[sPos])
				sPos++
				nnzElements++
			} else if s.colsIndex[sPos] > other.colsIndex[otherPos] {
				out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
				out.nzElements = append(out.nzElements, -other.nzElements[otherPos])
				otherPos++
				nnzElements++
			} else if s.colsIndex[sPos] == other.colsIndex[otherPos] {
				var result = s.nzElements[sPos] - other.nzElements[otherPos]
				if result != 0.0 {
					out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
					out.nzElements = append(out.nzElements, result)
					nnzElements++
				}
				sPos++
				otherPos++
			}
		}

		for sPos < s.nnzRow[i+1] {
			out.colsIndex = append(out.colsIndex, s.colsIndex[sPos])
			out.nzElements = append(out.nzElements, s.nzElements[sPos])
			sPos++
			nnzElements++
		}
		for otherPos < other.nnzRow[i+1] {
			out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
			out.nzElements = append(out.nzElements, -other.nzElements[otherPos])
			otherPos++
			nnzElements++

		}
		out.nnzRow[i+1] = nnzElements
	}
	return out
}

func (s *Sparse) prodSparse(other *Sparse) *Sparse {
	out := NewEmptySparse(s.rows, s.cols)
	var nnzElements = 0
	for i := 0; i < s.rows; i++ {
		var sPos, otherPos = s.nnzRow[i], other.nnzRow[i]
		out.nnzRow[0] = 0
		for sPos < s.nnzRow[i+1] && otherPos < other.nnzRow[i+1] {
			if s.colsIndex[sPos] == other.colsIndex[otherPos] {
				var result = other.nzElements[otherPos] * s.nzElements[sPos]
				if result != 0.0 {
					out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
					out.nzElements = append(out.nzElements, result)
					nnzElements++
				}
			}
			sPos++
			otherPos++
		}
		out.nnzRow[i+1] = nnzElements
	}
	return out
}

// Add returns the addition between the receiver and another matrix.
// It returns the same type of matrix of other, that is, a Dense matrix if
// other is Dense, or a Sparse matrix otherwise.
func (s *Sparse) Add(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat32: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense: // return dense
		out := other.Clone()
		s.DoNonZero(func(i, j int, v Float) {
			out.Data()[i*s.cols+j] += v
		})
		return out
	case *Sparse: // return sparse
		return s.addSparse(other)
	}
	return nil
}

// AddInPlace performs the in-place addition with the other matrix.
// It panics if other is not a Sparse matrix.
func (s *Sparse) AddInPlace(other Matrix) Matrix {
	switch other := other.(type) {
	case *Sparse:
		result := s.addSparse(other)
		s.colsIndex = result.colsIndex
		s.nnzRow = result.nnzRow
		s.nzElements = result.nzElements
	default:
		panic("mat32: AddInPlace(Dense) not implemented for Sparse matrices")
	}
	return s
}

// Sub returns the subtraction of the other matrix from the receiver.
// It returns a Dense matrix if other is Dense, or a Sparse matrix otherwise.
func (s *Sparse) Sub(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat32: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense: // return dense (not recommended)
		out := other.ProdScalar(-1.0)
		s.DoNonZero(func(i, j int, v Float) {
			out.Data()[i*s.cols+j] += v
		})
		return out
	case *Sparse: // return sparse
		return s.subSparse(other)
	}
	return nil
}

// SubInPlace performs the in-place subtraction with the other matrix.
// It panics if other is not a Sparse matrix.
func (s *Sparse) SubInPlace(other Matrix) Matrix {
	switch other := other.(type) {
	case *Sparse:
		result := s.subSparse(other)
		s.colsIndex = result.colsIndex
		s.nnzRow = result.nnzRow
		s.nzElements = result.nzElements
	default:
		panic("mat32: SubInPlace(Dense) not implemented for Sparse matrices")
	}
	return s
}

// Prod performs the element-wise product between the receiver and the other matrix,
// returning a new Sparse matrix.
func (s *Sparse) Prod(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat32: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense: // return sparse
		out := NewEmptySparse(other.rows, other.cols)
		copy(out.nnzRow, s.nnzRow)
		s.DoNonZero(func(i, j int, v Float) {
			var product = v * other.Data()[i*other.cols+j]
			if product != 0.0 {
				out.nzElements = append(out.nzElements, product)
				out.colsIndex = append(out.colsIndex, j)
			}
		})
		return out
	case *Sparse: // return sparse
		return s.prodSparse(other)
	}
	return nil
}

// ProdInPlace performs the in-place element-wise product with the other matrix.
// It panics if other is not a Sparse matrix.
func (s *Sparse) ProdInPlace(other Matrix) Matrix {
	switch other := other.(type) {
	case *Sparse:
		result := s.prodSparse(other)
		s.colsIndex = result.colsIndex
		s.nnzRow = result.nnzRow
		s.nzElements = result.nzElements
	default:
		panic("mat32: ProdInPlace(Dense) not implemented for Sparse matrices")
	}
	return s
}

// Div returns the result of the element-wise division of the receiver by the other matrix,
// returning a new Sparse matrix.
// It panics if other is a Sparse matrix.
func (s *Sparse) Div(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat32: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense: // return sparse?
		out := NewEmptySparse(other.rows, other.cols)
		copy(out.nnzRow, s.nnzRow)
		s.DoNonZero(func(i, j int, v Float) {
			var division = v / other.Data()[i*other.cols+j]
			if division != 0.0 {
				out.nzElements = append(out.nzElements, division)
				out.colsIndex = append(out.colsIndex, j)
			}
		})
		return out
	default: // TODO: return sparse?
		panic("mat32: Not permitted")
	}
}

// DivInPlace is currently not implemented for a Sparse matrix (it always panics).
func (s *Sparse) DivInPlace(other Matrix) Matrix {
	panic("mat32: DivInPlace not implemented for Sparse matrices")
}

// Mul performs the multiplication row by column, returning a Dense matrix.
// If A is an i×j Matrix, and B is j×k, then the resulting Matrix C = AB will be i×k.
func (s *Sparse) Mul(other Matrix) Matrix {
	if s.Columns() != other.Rows() {
		panic("mat32: matrices with not compatible size")
	}
	out := GetEmptyDenseWorkspace(s.Rows(), other.Columns())

	switch b := other.(type) {
	case *Dense:
		s.DoNonZero(func(i, j int, v Float) {
			for k := 0; k < b.cols; k++ {
				out.data[i*b.cols+k] += v * b.data[j*b.cols+k]
			}
		})
	case *Sparse:
		if b.IsVector() {
			s.DoNonZero(func(i, j int, v Float) {
				for k := 0; k < b.cols; k++ {
					out.data[i*b.cols+k] += v * b.AtVec(j)
				}
			})
		} else {
			s.DoNonZero(func(i, j int, v Float) {
				for k := 0; k < b.cols; k++ {
					out.data[i*b.cols+k] += v * b.At(j, k)
				}
			})
		}
	}
	return out
}

// DotUnitary returns the dot product of two vectors.
func (s *Sparse) DotUnitary(other Matrix) Float {
	if s.Size() != other.Size() {
		panic("mat32: incompatible sizes.")
	}
	var sum Float = 0.0
	switch b := other.(type) {
	case *Dense:
		s.DoNonZero(func(i, j int, v Float) {
			sum += b.Data()[i*b.cols+j] * v
		})
	case *Sparse:
		for i := 0; i < s.rows; i++ {
			var sPos, otherPos = s.nnzRow[i], b.nnzRow[i]
			for sPos < s.nnzRow[i+1] && otherPos < b.nnzRow[i+1] {
				if s.colsIndex[sPos] == b.colsIndex[otherPos] {
					sum += b.nzElements[otherPos] * s.nzElements[sPos]
				}
				sPos++
				otherPos++
			}
		}
		return sum
	}
	return sum
}

// Pow returns a new matrix, applying the power function with given exponent to all elements
// of the matrix.
func (s *Sparse) Pow(power Float) Matrix {
	out := s.Clone().(*Sparse) // TODO: find a better alternative to s.Clone()
	for i := 0; i < len(s.nzElements); i++ {
		out.nzElements[i] = Float(math.Pow(float64(out.nzElements[i]), float64(power)))
	}
	return out
}

// Norm returns the vector's norm. Use pow = 2.0 to compute the Euclidean norm.
func (s *Sparse) Norm(pow Float) Float {
	var sum Float = 0.0
	for i := 0; i < len(s.nzElements); i++ {
		sum += Float(math.Pow(float64(s.nzElements[i]), float64(pow)))
	}
	norm := Float(math.Pow(float64(sum), float64(1/pow)))
	return norm
}

// Sqrt returns a new matrix applying the square root function to all elements.
func (s *Sparse) Sqrt() Matrix {
	out := s.Clone().(*Sparse) // TODO: find a better alternative to s.Clone()
	for i := 0; i < len(s.nzElements); i++ {
		out.nzElements[i] = Float(math.Sqrt(float64(out.nzElements[i])))
	}
	return out
}

// ClipInPlace clips in place each value of the matrix.
func (s *Sparse) ClipInPlace(min, max Float) Matrix {
	for i := 0; i < len(s.nzElements); i++ {
		if s.nzElements[i] < min {
			s.nzElements[i] = min
		} else if s.nzElements[i] > max {
			s.nzElements[i] = max
		}
	}
	return s
}

// SplitV is currently not implemented for a Sparse matrix (it always panics).
func (s *Sparse) SplitV(_ ...int) []Matrix {
	panic("mat32: SplitV not implemented for Sparse matrices")
}

// MulT is currently not implemented for a Sparse matrix (it always panics).
func (s *Sparse) MulT(_ Matrix) Matrix {
	panic("mat32: MulT not implemented for Sparse matrices")
}

// Inverse returns the inverse of the matrix.
func (s *Sparse) Inverse() Matrix {
	panic("mat32: Sparse not implemented for Sparse matrices")
}

// Abs returns a new matrix applying the absolute value function to all elements.
func (s *Sparse) Abs() Matrix {
	out := s.Clone().(*Sparse) // TODO: find a better alternative to s.Clone()
	for i := 0; i < len(s.nzElements); i++ {
		out.nzElements[i] = Float(math.Abs(float64(out.nzElements[i])))
	}
	return out
}

// Sum returns the sum of all values of the matrix.
func (s *Sparse) Sum() Float {
	var sum Float = 0.0
	for i := 0; i < len(s.nzElements); i++ {
		sum += s.nzElements[i]
	}
	return sum
}

// Max returns the maximum value of the matrix.
func (s *Sparse) Max() Float {
	max := Float(math.Inf(-1))
	for _, v := range s.nzElements {
		if v > max {
			max = v
		}
	}
	return max
}

// Min returns the minimum value of the matrix.
func (s *Sparse) Min() Float {
	min := Float(math.Inf(1))
	for _, v := range s.nzElements {
		if v < min {
			min = v
		}
	}
	return min
}

// String returns a string representation of the matrix data.
func (s *Sparse) String() string {
	return fmt.Sprintf("%v", s.ToDense().data)
}

// SetData is currently not implemented for a Sparse matrix (it always panics).
func (s *Sparse) SetData(data []Float) {
	panic("mat32: SetData not implemented for Sparse matrices")
}

func (s *Sparse) maximumSparse(other *Sparse) *Sparse {
	out := NewEmptySparse(s.rows, s.cols)
	var nnzElements = 0
	for i := 0; i < s.rows; i++ {
		var sPos, otherPos = s.nnzRow[i], other.nnzRow[i]
		out.nnzRow[0] = 0
		for sPos < s.nnzRow[i+1] && otherPos < other.nnzRow[i+1] {
			if s.colsIndex[sPos] < other.colsIndex[otherPos] {
				if s.nzElements[sPos] > 0.0 {
					out.colsIndex = append(out.colsIndex, s.colsIndex[sPos])
					out.nzElements = append(out.nzElements, s.nzElements[sPos])
					nnzElements++
				}
				sPos++
			} else if s.colsIndex[sPos] > other.colsIndex[otherPos] {
				if other.nzElements[sPos] > 0.0 {
					out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
					out.nzElements = append(out.nzElements, other.nzElements[otherPos])
					nnzElements++
				}
				otherPos++
			} else if s.colsIndex[sPos] == other.colsIndex[otherPos] {
				var result = other.nzElements[otherPos] > s.nzElements[sPos]
				if result {
					out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
					out.nzElements = append(out.nzElements, other.nzElements[otherPos])
					nnzElements++
				} else {
					out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
					out.nzElements = append(out.nzElements, s.nzElements[otherPos])
					nnzElements++
				}
				sPos++
				otherPos++
			}
		}

		for sPos < s.nnzRow[i+1] {
			if s.nzElements[sPos] > 0.0 {
				out.colsIndex = append(out.colsIndex, s.colsIndex[sPos])
				out.nzElements = append(out.nzElements, s.nzElements[sPos])
				nnzElements++
			}
			sPos++
		}
		for otherPos < other.nnzRow[i+1] {
			if other.nzElements[otherPos] > 0.0 {
				out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
				out.nzElements = append(out.nzElements, other.nzElements[otherPos])
				nnzElements++
			}
			otherPos++
		}
		out.nnzRow[i+1] = nnzElements
	}
	return out
}

func (s *Sparse) minimumSparse(other *Sparse) *Sparse {
	out := NewEmptySparse(s.rows, s.cols)
	var nnzElements = 0
	for i := 0; i < s.rows; i++ {
		var sPos, otherPos = s.nnzRow[i], other.nnzRow[i]
		out.nnzRow[0] = 0
		for sPos < s.nnzRow[i+1] && otherPos < other.nnzRow[i+1] {
			if s.colsIndex[sPos] < other.colsIndex[otherPos] {
				if s.nzElements[sPos] < 0.0 {
					out.colsIndex = append(out.colsIndex, s.colsIndex[sPos])
					out.nzElements = append(out.nzElements, s.nzElements[sPos])
					nnzElements++
				}
				sPos++
			} else if s.colsIndex[sPos] > other.colsIndex[otherPos] {
				if other.nzElements[sPos] < 0.0 {
					out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
					out.nzElements = append(out.nzElements, other.nzElements[otherPos])
					nnzElements++
				}
				otherPos++
			} else if s.colsIndex[sPos] == other.colsIndex[otherPos] {
				var result = other.nzElements[otherPos] < s.nzElements[sPos]
				if result {
					out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
					out.nzElements = append(out.nzElements, other.nzElements[otherPos])
					nnzElements++
				} else {
					out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
					out.nzElements = append(out.nzElements, s.nzElements[otherPos])
					nnzElements++
				}
				sPos++
				otherPos++
			}
		}

		for sPos < s.nnzRow[i+1] {
			if s.nzElements[sPos] < 0.0 {
				out.colsIndex = append(out.colsIndex, s.colsIndex[sPos])
				out.nzElements = append(out.nzElements, s.nzElements[sPos])
				nnzElements++
			}
			sPos++
		}
		for otherPos < other.nnzRow[i+1] {
			if other.nzElements[otherPos] < 0.0 {
				out.colsIndex = append(out.colsIndex, other.colsIndex[otherPos])
				out.nzElements = append(out.nzElements, other.nzElements[otherPos])
				nnzElements++
			}
			otherPos++
		}
		out.nnzRow[i+1] = nnzElements
	}
	return out
}

// Maximum returns a new Sparse matrix initialized with the element-wise
// maximum value between the receiver and the other matrix.
func (s *Sparse) Maximum(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat32: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Sparse:
		return s.maximumSparse(other)
	default:
		panic("mat32: Maximum not implemented between Dense and Sparse matrices")
	}
}

// Minimum returns a new Sparse matrix initialized with the element-wise
// minimum value between the receiver and the other matrix.
func (s *Sparse) Minimum(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat32: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Sparse:
		return s.minimumSparse(other)
	default:
		panic("mat32: Minimum not implemented between Dense and Sparse matrices")
	}
}
