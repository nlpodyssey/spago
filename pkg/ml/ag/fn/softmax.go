// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function = &Softmax{}

// Softmax is a single-input softmax function.
type Softmax struct {
	x Operand
	y mat.Matrix[mat.Float] // initialized during the forward pass (required by the backward pass)
}

// NewSoftmax returns a new Softmax Function.
func NewSoftmax(x Operand) *Softmax {
	return &Softmax{x: x}
}

// Forward computes the output of this function.
func (r *Softmax) Forward() mat.Matrix[mat.Float] {
	r.y = mat.NewVecDense(softmax(r.x.Value().Data()))
	return r.y
}

// Backward computes the backward pass.
func (r *Softmax) Backward(gy mat.Matrix[mat.Float]) {
	if !(r.x.Value().SameDims(gy) || r.x.Value().VectorOfSameSize(gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		n := r.y.Size()
		jb := mat.GetDensePool[mat.Float]().Get(n, n)
		defer mat.ReleaseDense(jb)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i == j {
					jb.Set(i, j, r.y.AtVec(i)*(1.0-r.y.AtVec(j)))
				} else {
					jb.Set(i, j, -r.y.AtVec(i)*r.y.AtVec(j))
				}
			}
		}
		gx := jb.Mul(gy)
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}

func max(v []mat.Float) (m mat.Float) {
	m = v[len(v)-1]
	for _, e := range v {
		if m <= e {
			m = e
		}
	}
	return
}

func softmax(v []mat.Float) []mat.Float {
	maximum := max(v)
	var sum mat.Float = 0.0
	out := make([]mat.Float, len(v))
	for i, x := range v {
		e := mat.Exp(x - maximum)
		out[i] = e
		sum += e
	}
	for i := range v {
		out[i] /= sum
	}
	return out
}
