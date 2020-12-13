// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"math"
)

var _ Matrix = &Sparse{}

type Sparse struct {
	rows       int
	cols       int
	size       int       // rows*cols
	nzElements []float64 // A vector
	nnzRow     []int     // IA vector
	colsIndex  []int     // JA vector
}

// NewSparse returns a new rows x cols sparse matrix populated with a copy of the non-zero elements.
// The elements cannot be nil, panic otherwise. Use NewEmptySparse to initialize an empty matrix.
func NewSparse(rows, cols int, elements []float64) *Sparse {
	if elements == nil {
		panic("mat: elements cannot be nil. Use NewEmptySparse() instead.")
	}
	if len(elements) != rows*cols {
		panic(fmt.Sprintf("mat: wrong matrix dimensions. Elements size must be: %d", rows*cols))
	}
	return newSparse(rows, cols, elements)
}

// NewVecSparse returns a new column sparse vector populated with the non-zero elements.
// The elements cannot be nil, panic otherwise. Use NewEmptyVecSparse to initialize an empty matrix.
func NewVecSparse(elements []float64) *Sparse {
	if elements == nil {
		panic("mat: elements cannot be nil. Use NewEmptyVecSparse() instead.")
	}
	return newSparse(len(elements), 1, elements)
}

// NewEmptyVecSparse returns a new sparse vector of the given size.
func NewEmptyVecSparse(size int) *Sparse {
	return NewEmptySparse(size, 1)
}

// NewEmptySparse returns a new rows x cols Sparse matrix.
func NewEmptySparse(rows, cols int) *Sparse {
	return newSparse(rows, cols, make([]float64, rows*cols))
}

func newSparse(rows, cols int, elements []float64) *Sparse {
	nzElements := make([]float64, 0)
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

type Coordinate struct {
	I, J int
}

func NewSparseFromMap(rows, cols int, elements map[Coordinate]float64) *Sparse {
	nzElements := make([]float64, 0)
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

func OneHotSparse(size int, oneAt int) *Sparse {
	if oneAt >= size {
		panic(fmt.Sprintf("mat: impossible to set the one at index %d. The size is: %d", oneAt, size))
	}
	vec := NewEmptyVecSparse(size)
	vec.nzElements = append(vec.nzElements, 1.0)
	vec.colsIndex = append(vec.colsIndex, 0)
	for i := oneAt + 1; i < size; i++ {
		vec.nnzRow[i] = 1
	}
	return vec
}

func (s *Sparse) Sparsity() float64 {
	return float64(s.size-len(s.nzElements)) / float64(s.size)
}

func (s *Sparse) ToDense() *Dense {
	out := NewEmptyDense(s.rows, s.cols)
	for i := 0; i < s.rows; i++ {
		for elem := s.nnzRow[i]; elem < s.nnzRow[i+1]; elem++ {
			out.data[i*s.cols+s.colsIndex[elem]] = s.nzElements[elem]
		}
	}
	return out
}

func (s *Sparse) ZerosLike() Matrix {
	return NewEmptySparse(s.Dims())
}

func (s *Sparse) OnesLike() Matrix {
	panic("mat: OnesLike not implemented for Sparse matrices")
}

func (s *Sparse) Clone() Matrix {
	return NewSparse(s.rows, s.cols, s.Data())
}

func (s *Sparse) Copy(other Matrix) {
	if !SameDims(s, other) {
		panic("mat: incompatible matrix dimensions.")
	}
	if other, ok := other.(*Sparse); !ok {
		panic("mat: incompatible matrix types.")
	} else {
		s.colsIndex = append(s.colsIndex[:0], other.colsIndex...)
		s.nnzRow = append(s.nnzRow[:0], other.nnzRow...)
		s.nzElements = append(s.nzElements[:0], other.nzElements...)
	}
}

func (s *Sparse) Zeros() {
	s.nzElements = make([]float64, 0)
	s.nnzRow = make([]int, s.rows+1)
	s.nnzRow[0] = 0
	s.colsIndex = make([]int, 0)
}

func (s *Sparse) Dims() (r, c int) {
	return s.rows, s.cols
}

func (s *Sparse) Rows() int {
	return s.rows
}

func (s *Sparse) Columns() int {
	return s.cols
}

func (s *Sparse) Size() int {
	return s.size
}

func (s *Sparse) LastIndex() int {
	return s.size - 1
}

func (s *Sparse) Data() []float64 {
	out := make([]float64, s.rows*s.cols)
	for i := 0; i < s.rows; i++ {
		for elem := s.nnzRow[i]; elem < s.nnzRow[i+1]; elem++ {
			out[i*s.cols+s.colsIndex[elem]] = s.nzElements[elem]
		}
	}
	return out
}

func (s *Sparse) IsVector() bool {
	return s.rows == 1 || s.cols == 1
}

func (s *Sparse) IsScalar() bool {
	return s.size == 1
}

func (s *Sparse) Scalar() float64 {
	if !s.IsScalar() {
		panic("mat: expected scalar but the matrix contains more elements.")
	}
	if len(s.nzElements) > 0 {
		return s.nzElements[0]
	}
	return 0.0
}

func (s *Sparse) Set(i int, j int, v float64) {
	panic("mat: Set not implemented for Sparse matrices")
}

func (s *Sparse) At(i int, j int) float64 {
	if i >= s.rows {
		panic("mat: 'i' argument out of range.")
	}
	if j >= s.cols {
		panic("mat: 'j' argument out of range")
	}
	for k := s.nnzRow[i]; k < s.nnzRow[i+1]; k++ {
		if j == s.colsIndex[k] {
			return s.nzElements[k]
		}
	}
	return 0.0
}

func (s *Sparse) SetVec(i int, v float64) {
	panic("mat: SetVec not implemented for Sparse matrices")
}

func (s *Sparse) AtVec(i int) float64 {
	if !(s.IsVector()) {
		panic("mat: expected vector")
	}
	if i >= s.size {
		panic("mat: 'i' argument out of range.")
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

func (s *Sparse) DoNonZero(fn func(i, j int, v float64)) {
	for i := 0; i < s.rows; i++ {
		for elem := s.nnzRow[i]; elem < s.nnzRow[i+1]; elem++ {
			j := s.colsIndex[elem]
			v := s.nzElements[elem]
			fn(i, j, v)
		}
	}
}

func (s *Sparse) T() Matrix {
	// Convert CSR to CSC
	out := NewEmptySparse(s.cols, s.rows)
	out.nzElements = make([]float64, len(s.nzElements))
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
	s.DoNonZero(func(i, j int, v float64) {
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

func (s *Sparse) Reshape(r, c int) Matrix {
	panic("mat: Reshape not implemented for Sparse matrices")
}

// Apply executes the unary function fn.
// Important: apply to Functions such that f(0) = 0 (i.e. Sin, Tan)
func (s *Sparse) Apply(fn func(i, j int, v float64) float64, a Matrix) {
	if _, ok := a.(*Sparse); !ok {
		panic("mat: incompatible matrix types.")
	}
	for i := 0; i < len(s.nzElements); i++ {
		s.nzElements[i] = fn(i, 0, a.(*Sparse).nzElements[i])
	}
}

// ApplyWithAlpha executes the unary function fn, taking additional parameters alpha.
// It is currently not implemented for a Sparse matrix.
// Important: apply to Functions such that f(0, a) = 0
func (s *Sparse) ApplyWithAlpha(fn func(i, j int, v float64, alpha ...float64) float64, a Matrix, alpha ...float64) {
	panic("mat: ApplyWithAlpha not implemented for Sparse matrices")
}

// AddScalar performs an addition between the Sparse matrix and a float,
// returning a new Dense matrix.
func (s *Sparse) AddScalar(n float64) Matrix {
	out := NewInitDense(s.rows, s.cols, n)
	s.DoNonZero(func(i, j int, v float64) {
		out.Data()[i*s.cols+j] += v
	})

	return out
}

// AddScalarInPlace adds the scalar to the receiver.
// It is currently not implemented for a Sparse matrix.
func (s *Sparse) AddScalarInPlace(n float64) Matrix {
	panic("mat: AddScalarInPlace not implemented for Sparse matrices")
}

// SubScalar performs a subtraction between the Sparse Matrix and a float,
// returning a new Dense matrix.
func (s *Sparse) SubScalar(n float64) Matrix {
	out := NewInitDense(s.rows, s.cols, -n)
	s.DoNonZero(func(i, j int, v float64) {
		out.Data()[i*s.cols+j] += v
	})
	return out
}

// SubScalarInPlace subtracts the scalar to the receiver.
// It is currently not implemented for a Sparse matrix.
func (s *Sparse) SubScalarInPlace(n float64) Matrix {
	panic("mat: SubScalarInPlace not implemented for Sparse matrices")
}

// ProdScalar returns the multiplication of the float with the receiver,
// returning a new Sparse matrix.
func (s *Sparse) ProdScalar(n float64) Matrix {
	out := s.Clone().(*Sparse) // TODO: find a better alternative to s.Clone()
	if n == 0.0 {
		return NewEmptySparse(s.rows, s.cols)
	}
	for i, elem := range s.nzElements {
		out.nzElements[i] = elem * n
	}
	return out
}

// ProdScalarInPlace multiplies a float with the receiver in place,
// returning the same receiver Sparse matrix.
func (s *Sparse) ProdScalarInPlace(n float64) Matrix {
	if n == 0.0 {
		return NewEmptySparse(s.rows, s.cols)
	}
	for i, elem := range s.nzElements {
		s.nzElements[i] = elem * n
	}
	return s
}

// ProdMatrixScalarInPlace multiplies a matrix with a float, storing the result
// in the receiver, and returning the same receiver Sparse matrix.
func (s *Sparse) ProdMatrixScalarInPlace(m Matrix, n float64) Matrix {
	if _, ok := m.(*Sparse); !ok {
		panic("mat: incompatible matrix types.")
	}
	if !SameDims(s, m) {
		panic("mat: incompatible matrix dimensions.")
	}
	if n == 0.0 {
		return NewEmptySparse(s.rows, s.cols)
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

// Add returns the addition of a matrix with the receiver. It returns the same
// type of matrix of the other, that is a Dense matrix if other is Dense,
// or a Sparse matrix otherwise.
// It returns a Dense matrix if other is Dense, or a Sparse matrix otherwise.
func (s *Sparse) Add(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense: // return dense
		out := other.Clone()
		s.DoNonZero(func(i, j int, v float64) {
			out.Data()[i*s.cols+j] += v
		})
		return out
	case *Sparse: // return sparse
		return s.addSparse(other)
	}
	return nil
}

func (s *Sparse) AddInPlace(other Matrix) Matrix {
	switch other := other.(type) {
	case *Sparse:
		result := s.addSparse(other)
		s.colsIndex = result.colsIndex
		s.nnzRow = result.nnzRow
		s.nzElements = result.nzElements
	default:
		panic("mat: AddInPlace(Dense) not implemented for Sparse matrices")
	}
	return s
}

// Sub returns the subtraction of a matrix from the receiver.
// It returns a Dense matrix if other is Dense, or a Sparse matrix otherwise.
func (s *Sparse) Sub(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense: // return dense (not recommended)
		out := other.ProdScalar(-1.0)
		s.DoNonZero(func(i, j int, v float64) {
			out.Data()[i*s.cols+j] += v
		})
		return out
	case *Sparse: // return sparse
		return s.subSparse(other)
	}
	return nil
}

func (s *Sparse) SubInPlace(other Matrix) Matrix {
	switch other := other.(type) {
	case *Sparse:
		result := s.subSparse(other)
		s.colsIndex = result.colsIndex
		s.nnzRow = result.nnzRow
		s.nzElements = result.nzElements
	default:
		panic("mat: SubInPlace(Dense) not implemented for Sparse matrices")
	}
	return s
}

// Prod performs the element-wise product with the receiver,
// returning a new Sparse matrix.
func (s *Sparse) Prod(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense: // return sparse
		out := NewEmptySparse(other.rows, other.cols)
		copy(out.nnzRow, s.nnzRow)
		s.DoNonZero(func(i, j int, v float64) {
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

func (s *Sparse) ProdInPlace(other Matrix) Matrix {
	switch other := other.(type) {
	case *Sparse:
		result := s.prodSparse(other)
		s.colsIndex = result.colsIndex
		s.nnzRow = result.nnzRow
		s.nzElements = result.nzElements
	default:
		panic("mat: ProdInPlace(Dense) not implemented for Sparse matrices")
	}
	return s
}

// Div returns the result of the element-wise division,
// returning a new Sparse matrix.
func (s *Sparse) Div(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense: // return sparse?
		out := NewEmptySparse(other.rows, other.cols)
		copy(out.nnzRow, s.nnzRow)
		s.DoNonZero(func(i, j int, v float64) {
			var division = v / other.Data()[i*other.cols+j]
			if division != 0.0 {
				out.nzElements = append(out.nzElements, division)
				out.colsIndex = append(out.colsIndex, j)
			}
		})
		return out
	case *Sparse: // return sparse?
		panic("mat: Not permitted")
	}
	return nil
}

func (s *Sparse) DivInPlace(other Matrix) Matrix {
	panic("mat: DivInPlace not implemented for Sparse matrices")
}

// Mul performs the multiplication row by column, returning a Dense matrix.
func (s *Sparse) Mul(other Matrix) Matrix {
	if s.Columns() != other.Rows() {
		panic("mat: matrices with not compatible size")
	}
	out := GetEmptyDenseWorkspace(s.Rows(), other.Columns())

	switch b := other.(type) {
	case *Dense:
		s.DoNonZero(func(i, j int, v float64) {
			for k := 0; k < b.cols; k++ {
				var denseValue = b.data[j*b.cols+k]
				out.data[i*b.cols+k] += denseValue * v
			}
		})
		return out
	case *Sparse:
		s.DoNonZero(func(i, j int, v float64) {
			for k := 0; k < b.cols; k++ {
				var secondValue float64
				if b.IsVector() {
					secondValue = b.AtVec(j)
				} else {
					secondValue = b.At(j, k)
				}
				out.data[i*b.cols+k] += v * secondValue
			}
		})
		return out
	}
	return out
}

// DotUnitary returns the dot product of two vectors.
func (s *Sparse) DotUnitary(other Matrix) float64 {
	if s.Size() != other.Size() {
		panic("mat: incompatible sizes.")
	}
	sum := 0.0
	switch b := other.(type) {
	case *Dense:
		s.DoNonZero(func(i, j int, v float64) {
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

func (s *Sparse) Pow(power float64) Matrix {
	out := s.Clone().(*Sparse) // TODO: find a better alternative to s.Clone()
	for i := 0; i < len(s.nzElements); i++ {
		out.nzElements[i] = math.Pow(out.nzElements[i], power)
	}
	return out
}

func (s *Sparse) Norm(pow float64) float64 {
	sum := 0.0
	for i := 0; i < len(s.nzElements); i++ {
		sum += math.Pow(s.nzElements[i], pow)
	}
	norm := math.Pow(sum, 1/pow)
	return norm
}

func (s *Sparse) Sqrt() Matrix {
	out := s.Clone().(*Sparse) // TODO: find a better alternative to s.Clone()
	for i := 0; i < len(s.nzElements); i++ {
		out.nzElements[i] = math.Sqrt(out.nzElements[i])
	}
	return out
}

func (s *Sparse) ClipInPlace(min, max float64) Matrix {
	for i := 0; i < len(s.nzElements); i++ {
		if s.nzElements[i] < min {
			s.nzElements[i] = min
		} else if s.nzElements[i] > max {
			s.nzElements[i] = max
		}
	}
	return s
}

func (s *Sparse) Abs() Matrix {
	out := s.Clone().(*Sparse) // TODO: find a better alternative to s.Clone()
	for i := 0; i < len(s.nzElements); i++ {
		out.nzElements[i] = math.Abs(out.nzElements[i])
	}
	return out
}

func (s *Sparse) Sum() float64 {
	sum := 0.0
	for i := 0; i < len(s.nzElements); i++ {
		sum += s.nzElements[i]
	}
	return sum
}

func (s *Sparse) Max() float64 {
	max := math.Inf(-1)
	for _, v := range s.nzElements {
		if v > max {
			max = v
		}
	}
	return max
}

func (s *Sparse) Min() float64 {
	min := math.Inf(1)
	for _, v := range s.nzElements {
		if v < min {
			min = v
		}
	}
	return min
}

func (s *Sparse) String() string {
	return fmt.Sprintf("%v", s.ToDense().data)
}

func (s *Sparse) SetData(data []float64) {
	panic("mat: SetData not implemented for Sparse matrices")
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

func (s *Sparse) Maximum(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense: // return dense
		panic("mat: Maximum not implemented between Dense and Sparse matrices")
	case *Sparse: // return sparse
		return s.maximumSparse(other)
	}
	return nil
}

func (s *Sparse) Minimum(other Matrix) Matrix {
	if !(SameDims(s, other) ||
		(other.Columns() == 1 && other.Rows() == s.Rows()) ||
		(other.IsVector() && s.IsVector() && other.Size() == s.Size())) {
		panic("mat: matrices with not compatible size")
	}
	switch other := other.(type) {
	case *Dense: // return dense
		panic("mat: Minimum not implemented between Dense and Sparse matrices")
	case *Sparse: // return sparse
		return s.minimumSparse(other)
	}
	return nil
}
