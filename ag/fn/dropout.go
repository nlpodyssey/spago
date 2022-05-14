// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/mat/rand/bernulli"
)

// Dropout is an operator to perform elements dropout with a probability.
type Dropout[O Operand] struct {
	x        O
	prob     float64
	q        float64 // 1 - p
	randGen  *rand.LockedRand
	mask     mat.Matrix // filled during the forward
	operands []O
}

// NewDropout returns a new Dropout Function.
func NewDropout[O Operand](x O, p float64, randGen *rand.LockedRand) *Dropout[O] {
	return &Dropout[O]{
		x:        x,
		prob:     p,
		q:        1.0 - float64(p),
		randGen:  randGen,
		mask:     nil,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *Dropout[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *Dropout[O]) Forward() mat.Matrix {
	xv := r.x.Value()
	if r.q > 0.0 {
		r.mask = bernulli.Distribution(xv.Rows(), xv.Columns(), r.prob, r.randGen)
		r.mask.ProdScalarInPlace(1.0 / r.q)
	} else {
		r.mask = xv.ZerosLike()
	}
	return xv.Prod(r.mask)
}

// Backward computes the backward pass.
func (r *Dropout[O]) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	defer mat.ReleaseMatrix(r.mask)
	if r.x.RequiresGrad() {
		gx := gy.Prod(r.mask)
		r.x.AccGrad(gx)
	}
}
