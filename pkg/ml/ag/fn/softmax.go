// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/saientist/spago/pkg/mat"
	"math"
)

// Single-input, softmax function.
type Softmax struct {
	x Operand
	y mat.Matrix // initialized during the forward pass (required by the backward pass)
}

func NewSoftmax(x Operand) *Softmax {
	return &Softmax{x: x}
}

// Forward computes the output of this function.
func (r *Softmax) Forward() mat.Matrix {
	r.y = mat.NewVecDense(softmax(r.x.Value().Data()))
	return r.y
}

func (r *Softmax) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		n := r.y.Size()
		jb := mat.NewEmptyDense(n, n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i == j {
					jb.Set(r.y.At(i)*(1.0-r.y.At(j)), i, j)
				} else {
					jb.Set(-r.y.At(i)*r.y.At(j), i, j)
				}
			}
		}
		r.x.PropagateGrad(jb.Mul(gy))
	}
}

func max(v []float64) (m float64) {
	m = v[len(v)-1]
	for _, e := range v {
		if m <= e {
			m = e
		}
	}
	return
}

func softmax(v []float64) (sm []float64) {
	c := max(v)
	var sum float64 = 0
	for _, e := range v {
		sum += math.Exp(e - c)
	}
	sm = make([]float64, len(v))
	for i, v := range v {
		sm[i] = math.Exp(v-c) / sum
	}
	return sm
}
