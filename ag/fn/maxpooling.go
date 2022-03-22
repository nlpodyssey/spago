// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/utils"
)

// MaxPooling is an operator to perform max pooling.
type MaxPooling[T mat.DType, O Operand[T]] struct {
	x    O
	rows int
	cols int
	// initialized during the forward pass
	y       mat.Matrix[T]
	argmaxI [][]int
	argmaxJ [][]int
}

// NewMaxPooling returns a new MaxPooling Function.
func NewMaxPooling[T mat.DType, O Operand[T]](x O, r, c int) *MaxPooling[T, O] {
	return &MaxPooling[T, O]{
		x:       x,
		rows:    r,
		cols:    c,
		y:       nil,
		argmaxI: nil,
		argmaxJ: nil,
	}
}

// Operands returns the list of operands.
func (r *MaxPooling[T, O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *MaxPooling[T, O]) Forward() mat.Matrix[T] {
	if !(r.x.Value().Rows()%r.rows == 0 && r.x.Value().Columns()%r.cols == 0) {
		panic("fn: size mismatch")
	}

	r.y = mat.NewEmptyDense[T](r.x.Value().Rows()/r.rows, r.x.Value().Columns()/r.cols)
	r.argmaxI = utils.MakeIntMatrix(r.y.Dims()) // output argmax row index
	r.argmaxJ = utils.MakeIntMatrix(r.y.Dims()) // output argmax column index

	for row := 0; row < r.y.Rows(); row++ {
		for col := 0; col < r.y.Columns(); col++ {
			maximum := mat.SmallestNonzero[T]()
			for i := row * r.rows; i < (row*r.rows)+r.rows; i++ {
				for j := col * r.cols; j < (col*r.cols)+r.rows; j++ {
					val := r.x.Value().At(i, j)
					if val > maximum {
						maximum = val
						r.argmaxI[row][col] = i
						r.argmaxJ[row][col] = j
					}
				}
			}
			r.y.Set(row, col, maximum)
		}
	}

	return r.y
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
				gx.Set(rowi[col], rowj[col], gy.At(row, col))
			}
		}
		r.x.PropagateGrad(gx)
	}
}
