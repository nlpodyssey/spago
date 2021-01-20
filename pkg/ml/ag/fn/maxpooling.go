// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/utils"
)

var _ Function = &MaxPooling{}

// MaxPooling is an operator to perform max pooling.
type MaxPooling struct {
	x    Operand
	rows int
	cols int
	// initialized during the forward pass
	y       mat.Matrix
	argmaxI [][]int
	argmaxJ [][]int
}

// NewMaxPooling returns a new MaxPooling Function.
func NewMaxPooling(x Operand, r, c int) *MaxPooling {
	return &MaxPooling{
		x:       x,
		rows:    r,
		cols:    c,
		y:       nil,
		argmaxI: nil,
		argmaxJ: nil,
	}
}

// Forward computes the output of the function.
func (r *MaxPooling) Forward() mat.Matrix {
	if !(r.x.Value().Rows()%r.rows == 0 && r.x.Value().Columns()%r.cols == 0) {
		panic("fn: size mismatch")
	}

	r.y = mat.NewEmptyDense(r.x.Value().Rows()/r.rows, r.x.Value().Columns()/r.cols)
	r.argmaxI = utils.MakeIntMatrix(r.y.Dims()) // output argmax row index
	r.argmaxJ = utils.MakeIntMatrix(r.y.Dims()) // output argmax column index

	for row := 0; row < r.y.Rows(); row++ {
		for col := 0; col < r.y.Columns(); col++ {
			maximum := mat.SmallestNonzeroFloat
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
func (r *MaxPooling) Backward(gy mat.Matrix) {
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
