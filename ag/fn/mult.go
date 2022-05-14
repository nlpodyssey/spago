// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

// MulT is an operator to perform matrix-vector multiplication.
type MulT[O Operand] struct {
	x1       O // matrix
	x2       O // vector
	operands []O
}

// NewMulT returns a new MulT Function.
func NewMulT[O Operand](x1 O, x2 O) *MulT[O] {
	return &MulT[O]{
		x1:       x1,
		x2:       x2,
		operands: []O{x1, x2},
	}
}

// Forward computes the output of the function.
func (r *MulT[O]) Forward() mat.Matrix {
	return r.x1.Value().MulT(r.x2.Value())
}

// Operands returns the list of operands.
func (r *MulT[O]) Operands() []O {
	return r.operands
}

// Backward computes the backward pass.
func (r *MulT[O]) Backward(gy mat.Matrix) {
	//if !(r.x1.Value().Rows() == gy.Rows() && r.x2.Value().Columns() == gy.Columns()) {
	//	panic("fn: matrices with not compatible size")
	//}
	var wg sync.WaitGroup
	if r.x1.RequiresGrad() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			x2t := r.x2.Value().T()
			defer mat.ReleaseMatrix(x2t)
			gx := gy.Mul(x2t)
			defer mat.ReleaseMatrix(gx)
			r.x1.AccGrad(gx.T())
		}()
	}
	if r.x2.RequiresGrad() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			//r.x2.AccGrad(gy.T().MulT(r.x1).T()) // alternative method
			if gy.Columns() == 1 {
				gx := r.x1.Value().Mul(gy)
				defer mat.ReleaseMatrix(gx)
				r.x2.AccGrad(gx)
			} else {
				x1t := r.x1.Value()
				defer mat.ReleaseMatrix(x1t)
				gx := x1t.MulT(gy)
				defer mat.ReleaseMatrix(gx)
				r.x2.AccGrad(gx)
			}
		}()
	}
	wg.Wait()
}
