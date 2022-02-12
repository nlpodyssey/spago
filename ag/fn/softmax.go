// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &Softmax[float32]{}

// Softmax is a single-input softmax function.
type Softmax[T mat.DType] struct {
	x Operand[T]
	y mat.Matrix[T] // initialized during the forward pass (required by the backward pass)
}

// NewSoftmax returns a new Softmax Function.
func NewSoftmax[T mat.DType](x Operand[T]) *Softmax[T] {
	return &Softmax[T]{x: x}
}

// Forward computes the output of this function.
func (r *Softmax[T]) Forward() mat.Matrix[T] {
	r.y = mat.NewVecDense(softmax(r.x.Value().Data()))
	return r.y
}

// Backward computes the backward pass.
func (r *Softmax[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		n := r.y.Size()
		jb := mat.GetDensePool[T]().Get(n, n)
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

func max[T mat.DType](v []T) (m T) {
	m = v[len(v)-1]
	for _, e := range v {
		if m <= e {
			m = e
		}
	}
	return
}

func softmax[T mat.DType](v []T) []T {
	maximum := max(v)
	var sum T = 0.0
	out := make([]T, len(v))
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
