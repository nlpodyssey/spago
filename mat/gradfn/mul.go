// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

// Mul is an operator to perform matrix-vector multiplication.
type Mul[O mat.Tensor] struct {
	x1 O // matrix
	x2 O // vector
}

// NewMul returns a new Mul Function.
func NewMul[O mat.Tensor](x1 O, x2 O) *Mul[O] {
	return &Mul[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *Mul[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x1, r.x2}
}

// Forward computes the output of the function.
func (r *Mul[O]) Forward() (mat.Tensor, error) {
	return r.x1.Value().(mat.Matrix).Mul(r.x2.Value().(mat.Matrix)), nil
}

// Backward computes the backward pass.
func (r *Mul[O]) Backward(gy mat.Tensor) error {
	if !(r.x1.Value().Shape()[0] == gy.Shape()[0] && r.x2.Value().Shape()[1] == gy.Shape()[1]) {
		return fmt.Errorf("fn: matrices with not compatible size")
	}
	var wg sync.WaitGroup
	if r.x1.RequiresGrad() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			x2t := r.x2.Value().(mat.Matrix).T()
			gx := gy.(mat.Matrix).Mul(x2t)
			r.x1.AccGrad(gx)
		}()
	}
	if r.x2.RequiresGrad() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			//r.x2.AccGrad(gy.T().Mul(r.x1).T()) // alternative method
			if gy.Shape()[1] == 1 {
				gx := r.x1.Value().(mat.Matrix).MulT(gy.(mat.Matrix))
				r.x2.AccGrad(gx)
			} else {
				x1t := r.x1.Value().(mat.Matrix).T()
				gx := x1t.Mul(gy.(mat.Matrix))
				r.x2.AccGrad(gx)
			}
		}()
	}
	wg.Wait()
	return nil
}
