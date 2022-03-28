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
type Dropout[T mat.DType, O Operand[T]] struct {
	x        O
	prob     T
	q        T // 1 - p
	randGen  *rand.LockedRand[T]
	mask     mat.Matrix[T] // filled during the forward
	operands []O
}

// NewDropout returns a new Dropout Function.
func NewDropout[T mat.DType, O Operand[T]](x O, p T, randGen *rand.LockedRand[T]) *Dropout[T, O] {
	return &Dropout[T, O]{
		x:        x,
		prob:     p,
		q:        1.0 - p,
		randGen:  randGen,
		mask:     nil,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *Dropout[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *Dropout[T, O]) Forward() mat.Matrix[T] {
	if r.q > 0.0 {
		r.mask = bernulli.Distribution(r.x.Value().Rows(), r.x.Value().Columns(), r.prob, r.randGen)
		r.mask.ProdScalarInPlace(1.0 / r.q)
	} else {
		r.mask = r.x.Value().ZerosLike()
	}
	return r.x.Value().Prod(r.mask)
}

// Backward computes the backward pass.
func (r *Dropout[T, O]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	defer mat.ReleaseMatrix(r.mask)
	if r.x.RequiresGrad() {
		gx := gy.Prod(r.mask)
		r.x.PropagateGrad(gx)
	}
}
