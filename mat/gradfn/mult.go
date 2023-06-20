// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

// MulT is an operator to perform matrix-vector multiplication.
type MulT[O mat.Tensor] struct {
	x1 O // matrix
	x2 O // vector
}

// NewMulT returns a new MulT Function.
func NewMulT[O mat.Tensor](x1 O, x2 O) *MulT[O] {
	return &MulT[O]{
		x1: x1,
		x2: x2,
	}
}

// Forward computes the output of the function.
func (r *MulT[O]) Forward() (mat.Tensor, error) {
	return r.x1.Value().(mat.Matrix).MulT(r.x2.Value().(mat.Matrix)), nil
}

// Operands returns the list of operands.
func (r *MulT[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x1, r.x2}
}

// Backward computes the backward pass.
func (r *MulT[O]) Backward(gy mat.Tensor) error {
	//if !(r.x1.Value().Shape()[0] == gy.Shape()[0] && r.x2.Value().Shape()[1] == gy.Shape()[1]) {
	//	panic("fn: matrices with not compatible size")
	//}
	var wg sync.WaitGroup
	if r.x1.RequiresGrad() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			gx := gy.(mat.Matrix).Mul(r.x2.Value().(mat.Matrix).T())
			r.x1.AccGrad(gx.T())
		}()
	}
	if r.x2.RequiresGrad() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			//r.x2.AccGrad(gy.T().MulT(r.x1).T()) // alternative method
			if gy.Shape()[1] == 1 {
				gx := r.x1.Value().(mat.Matrix).Mul(gy.(mat.Matrix))
				r.x2.AccGrad(gx)
			} else {
				gx := r.x1.Value().(mat.Matrix).MulT(gy.(mat.Matrix))
				r.x2.AccGrad(gx)
			}
		}()
	}
	wg.Wait()
	return nil
}
