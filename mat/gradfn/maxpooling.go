// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"math"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

// MaxPooling is an operator to perform max pooling.
type MaxPooling[O mat.Tensor] struct {
	x    O
	rows int
	cols int
	// initialized during the forward pass
	y       mat.Matrix
	argmaxI [][]int
	argmaxJ [][]int
}

// NewMaxPooling returns a new MaxPooling Function.
func NewMaxPooling[O mat.Tensor](x O, r, c int) *MaxPooling[O] {
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
func (r *MaxPooling[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the function.
func (r *MaxPooling[O]) Forward() (mat.Tensor, error) {
	xv := r.x.Value().(mat.Matrix)
	if !(xv.Shape()[0]%r.rows == 0 && xv.Shape()[1]%r.cols == 0) {
		panic("fn: size mismatch")
	}

	r.y = xv.NewMatrix(mat.WithShape(xv.Shape()[0]/r.rows, xv.Shape()[1]/r.cols))
	r.argmaxI = makeIntMatrix(r.y.Shape()) // output argmax row index
	r.argmaxJ = makeIntMatrix(r.y.Shape()) // output argmax column index

	for row := 0; row < r.y.Shape()[0]; row++ {
		for col := 0; col < r.y.Shape()[1]; col++ {
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
			r.y.SetScalar(float.Interface(maximum), row, col)
		}
	}

	return r.y, nil
}

// makeIntMatrix returns a new 2-dimensional slice of int.
func makeIntMatrix(indices []int) [][]int {
	rows := indices[0]
	cols := indices[1]

	matrix := make([][]int, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]int, cols)
	}
	return matrix
}

// Backward computes the backward pass.
func (r *MaxPooling[O]) Backward(gy mat.Tensor) error {
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).ZerosLike()
		for row := 0; row < r.y.Shape()[0]; row++ {
			rowi := r.argmaxI[row]
			rowj := r.argmaxJ[row]
			for col := 0; col < r.y.Shape()[1]; col++ {
				gx.SetScalar(gy.(mat.Matrix).ScalarAt(row, col), rowi[col], rowj[col])
			}
		}
		r.x.AccGrad(gx)
	}
	return nil
}
