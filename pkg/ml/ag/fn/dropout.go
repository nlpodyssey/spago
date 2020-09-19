// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/mat/rand/bernulli"
)

var _ Function = &Dropout{}

// Dropout is an operator to perform elements dropout with a probability.
type Dropout struct {
	x       Operand
	prob    float64
	q       float64 // 1 - p
	randGen *rand.LockedRand
	mask    mat.Matrix // filled during the forward
}

func NewDropout(x Operand, p float64, randGen *rand.LockedRand) *Dropout {
	return &Dropout{
		x:       x,
		prob:    p,
		q:       1.0 - p,
		randGen: randGen,
		mask:    nil,
	}
}

// Forward computes the output of the function.
func (r *Dropout) Forward() mat.Matrix {
	if r.q > 0.0 {
		r.mask = bernulli.Distribution(r.x.Value().Rows(), r.x.Value().Columns(), r.prob, r.randGen)
		r.mask.ProdScalarInPlace(1.0 / r.q)
	} else {
		r.mask = r.x.Value().ZerosLike()
	}
	return r.x.Value().Prod(r.mask)
}

func (r *Dropout) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	defer mat.ReleaseDense(r.mask.(*mat.Dense))
	if r.x.RequiresGrad() {
		gx := gy.Prod(r.mask)
		r.x.PropagateGrad(gx)
	}
}
