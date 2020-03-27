// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rnd"
	"golang.org/x/exp/rand"
)

type Dropout struct {
	x      Operand
	prob   float64
	q      float64 // 1 - p
	source rand.Source
	mask   mat.Matrix // filled during the forward
}

func NewDropout(x Operand, p float64, source rand.Source) *Dropout {
	return &Dropout{
		x:      x,
		prob:   p,
		q:      1.0 - p,
		source: source,
		mask:   nil,
	}
}

// Forward computes the output of the function.
func (r *Dropout) Forward() mat.Matrix {
	if r.q > 0.0 {
		r.mask = rnd.Bernoulli(r.x.Value().Rows(), r.x.Value().Columns(), r.prob, r.source)
		r.mask.ProdScalarInPlace(1.0 / r.q)
	} else {
		r.mask = r.x.Value().ZerosLike()
	}
	return r.x.Value().Prod(r.mask)
}

func (r *Dropout) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		r.x.PropagateGrad(gy.Prod(r.mask))
	}
}
