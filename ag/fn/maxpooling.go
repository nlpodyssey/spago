// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// MaxPooling is an operator to perform max pooling.
type MaxPooling[T mat.DType, O Operand[T]] struct {
	x    O
	rows int
	cols int
	// initialized during the forward pass
	y        mat.Matrix[T]
	argmaxI  [][]int
	argmaxJ  [][]int
	operands []O
}

// NewMaxPooling returns a new MaxPooling Function.
func NewMaxPooling[T mat.DType, O Operand[T]](x O, r, c int) *MaxPooling[T, O] {
	return &MaxPooling[T, O]{
		x:        x,
		rows:     r,
		cols:     c,
		y:        nil,
		argmaxI:  nil,
		argmaxJ:  nil,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *MaxPooling[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *MaxPooling[T, O]) Forward() mat.Matrix[T] {
	xv := r.x.Value()
	if !(xv.Rows()%r.rows == 0 && xv.Columns()%r.cols == 0) {
		panic("fn: size mismatch")
	}

	r.y = mat.NewEmptyDense[T](xv.Rows()/r.rows, xv.Columns()/r.cols)
	r.argmaxI = makeIntMatrix(r.y.Dims()) // output argmax row index
	r.argmaxJ = makeIntMatrix(r.y.Dims()) // output argmax column index

	for row := 0; row < r.y.Rows(); row++ {
		for col := 0; col < r.y.Columns(); col++ {
			maximum := mat.SmallestNonzero[T]()
			for i := row * r.rows; i < (row*r.rows)+r.rows; i++ {
				for j := col * r.cols; j < (col*r.cols)+r.rows; j++ {
					val := xv.ScalarAt(i, j)
					if val > maximum {
						maximum = val
						r.argmaxI[row][col] = i
						r.argmaxJ[row][col] = j
					}
				}
			}
			r.y.SetScalar(row, col, maximum)
		}
	}

	return r.y
}

// makeIntMatrix returns a new 2-dimensional slice of int.
func makeIntMatrix(rows, cols int) [][]int {
	matrix := make([][]int, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]int, cols)
	}
	return matrix
}

// Backward computes the backward pass.
func (r *MaxPooling[T, O]) Backward(gy mat.Matrix[T]) {
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		for row := 0; row < r.y.Rows(); row++ {
			rowi := r.argmaxI[row]
			rowj := r.argmaxJ[row]
			for col := 0; col < r.y.Columns(); col++ {
				gx.SetScalar(rowi[col], rowj[col], gy.ScalarAt(row, col))
			}
		}
		r.x.AccGrad(gx)
	}
}
