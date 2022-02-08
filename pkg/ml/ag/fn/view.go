// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function = &View{}

// View is a function to extract a portion of a matrix.
type View struct {
	x  Operand
	sx int
	sy int
	lx int // x length
	ly int // y length
}

// NewView returns a new View Function.
func NewView(x Operand, sx, sy, lx, ly int) *View {
	return &View{x: x, sx: sx, sy: sy, lx: lx, ly: ly}
}

// Forward computes the output of the function.
func (r *View) Forward() mat.Matrix[mat.Float] {
	y := mat.NewEmptyDense[mat.Float](r.lx, r.ly)
	for i := 0; i < r.lx; i++ {
		for j := 0; j < r.ly; j++ {
			y.Set(i, j, r.x.Value().At(i+r.sx, j+r.sy))
		}
	}
	return y
}

// Backward computes the backward pass.
func (r *View) Backward(gy mat.Matrix[mat.Float]) {
	if !(gy.Rows() == r.lx && gy.Columns() == r.ly) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.NewEmptyDense[mat.Float](r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		for i := 0; i < r.lx; i++ {
			for j := 0; j < r.ly; j++ {
				gx.Set(i+r.sx, j+r.sy, gy.At(i, j))
			}
		}
		r.x.PropagateGrad(gx)
	}
}
