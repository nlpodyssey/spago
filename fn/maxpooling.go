// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"math"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

// MaxPooling is an operator to perform max pooling.
type MaxPooling[O DualValue] struct {
	x    O
	rows int
	cols int
	// initialized during the forward pass
	y       mat.Matrix
	argmaxI [][]int
	argmaxJ [][]int
}

// NewMaxPooling returns a new MaxPooling Function.
func NewMaxPooling[O DualValue](x O, r, c int) *MaxPooling[O] {
	return &MaxPooling[O]{
		x:       x,
		rows:    r,
		cols:    c,
		y:       nil,
		argmaxI: nil,
		argmaxJ: nil,
	}
}

// Operands returns the list of operands.
func (r *MaxPooling[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *MaxPooling[O]) Forward() mat.Matrix {
	xv := r.x.Value()
	if !(xv.Rows()%r.rows == 0 && xv.Columns()%r.cols == 0) {
		panic("fn: size mismatch")
	}

	r.y = xv.NewEmptyMatrix(xv.Rows()/r.rows, xv.Columns()/r.cols)
	r.argmaxI = makeIntMatrix(r.y.Dims()) // output argmax row index
	r.argmaxJ = makeIntMatrix(r.y.Dims()) // output argmax column index

	for row := 0; row < r.y.Rows(); row++ {
		for col := 0; col < r.y.Columns(); col++ {
			maximum := math.SmallestNonzeroFloat64

			maxRows := (row * r.rows) + r.rows
			for i := row * r.rows; i < maxRows; i++ {
				maxCols := (col * r.cols) + r.rows
				for j := col * r.cols; j < maxCols; j++ {
					// FIXME: avoid casting to specific type
					val := xv.ScalarAt(i, j).F64()
					if val > maximum {
						maximum = val
						r.argmaxI[row][col] = i
						r.argmaxJ[row][col] = j
					}
				}
			}
			r.y.SetScalar(row, col, float.Interface(maximum))
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
func (r *MaxPooling[O]) Backward(gy mat.Matrix) {
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
