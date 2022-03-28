// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"sync"
)

// Mul is an operator to perform matrix-vector multiplication.
type Mul[T mat.DType, O Operand[T]] struct {
	x1       O // matrix
	x2       O // vector
	operands []O
}

// NewMul returns a new Mul Function.
func NewMul[T mat.DType, O Operand[T]](x1 O, x2 O) *Mul[T, O] {
	return &Mul[T, O]{
		x1:       x1,
		x2:       x2,
		operands: []O{x1, x2},
	}
}

// Operands returns the list of operands.
func (r *Mul[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *Mul[T, O]) Forward() mat.Matrix[T] {
	if r.x1.Value().Columns() != r.x2.Value().Rows() {
		panic("fn: matrices with not compatible size")
	}
	return r.x1.Value().Mul(r.x2.Value())
}

// Backward computes the backward pass.
// TODO: backward of sparse gradients
func (r *Mul[T, O]) Backward(gy mat.Matrix[T]) {
	if !(r.x1.Value().Rows() == gy.Rows() && r.x2.Value().Columns() == gy.Columns()) {
		panic("fn: matrices with not compatible size")
	}
	var wg sync.WaitGroup
	if r.x1.RequiresGrad() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			x2t := r.x2.Value().T()
			defer mat.ReleaseMatrix(x2t)
			gx := gy.Mul(x2t)
			defer mat.ReleaseMatrix(gx)
			r.x1.PropagateGrad(gx)
		}()
	}
	if r.x2.RequiresGrad() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			//r.x2.PropagateGrad(gy.T().Mul(r.x1).T()) // alternative method
			if gy.Columns() == 1 {
				gx := r.x1.Value().MulT(gy)
				defer mat.ReleaseMatrix(gx)
				r.x2.PropagateGrad(gx)
			} else {
				x1t := r.x1.Value().T()
				defer mat.ReleaseMatrix(x1t)
				gx := x1t.Mul(gy)
				defer mat.ReleaseMatrix(gx)
				r.x2.PropagateGrad(gx)
			}
		}()
	}
	wg.Wait()
}
