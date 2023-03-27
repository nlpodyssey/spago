// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/mat/rand/bernulli"
)

// Dropout is an operator to perform elements dropout with a probability.
type Dropout[O DualValue] struct {
	x       O
	prob    float64
	q       float64 // 1 - p
	randGen *rand.LockedRand
	mask    mat.Matrix // filled during the forward
}

// NewDropout returns a new Dropout Function.
func NewDropout[O DualValue](x O, p float64, randGen *rand.LockedRand) *Dropout[O] {
	return &Dropout[O]{
		x:       x,
		prob:    p,
		q:       1.0 - float64(p),
		randGen: randGen,
		mask:    nil,
	}
}

// Operands returns the list of operands.
func (r *Dropout[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *Dropout[O]) Forward() (mat.Matrix, error) {
	xv := r.x.Value()
	if r.q > 0.0 {
		// FIXME: avoid casting to specific type
		r.mask = bernulli.Distribution[float64](xv.Rows(), xv.Columns(), r.prob, r.randGen)
		r.mask.ProdScalarInPlace(1.0 / r.q)
	} else {
		r.mask = xv.ZerosLike()
	}
	return xv.Prod(r.mask), nil
}

// Backward computes the backward pass.
func (r *Dropout[O]) Backward(gy mat.Matrix) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	defer mat.ReleaseMatrix(r.mask)
	if r.x.RequiresGrad() {
		gx := gy.Prod(r.mask)
		r.x.AccGrad(gx)
	}
	return nil
}
