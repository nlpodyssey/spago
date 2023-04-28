// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

// Max is an operator to perform element-wise max.
// y = max(x1, x2)
type Max[O DualValue] struct {
	x1 O
	x2 O
}

// NewMax returns a new Max Function.
func NewMax[O DualValue](x1 O, x2 O) *Max[O] {
	return &Max[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *Max[O]) Operands() []O {
	return []O{r.x1, r.x2}
}

// Forward computes the output of the function.
func (r *Max[O]) Forward() (mat.Matrix, error) {
	return r.x1.Value().Maximum(r.x2.Value()), nil
}

// Backward computes the backward pass.
func (r *Max[O]) Backward(gy mat.Matrix) error {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !mat.SameDims(x1v, gy) || !mat.SameDims(x2v, gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}

	n := gy.Size()
	// FIXME: avoid casting to specific type
	gyData := mat.Data[float64](gy)
	x1vData := mat.Data[float64](x1v)
	x2vData := mat.Data[float64](x2v)

	if r.x1.RequiresGrad() {
		gxData := make([]float64, n)
		for i := 0; i < n; i++ {
			if x1vData[i] > x2vData[i] {
				gxData[i] = gyData[i]
			}
		}
		gx := x1v.NewVec(float.SliceInterface(gxData))
		r.x1.AccGrad(gx)
	}
	if r.x2.RequiresGrad() {
		gxData := make([]float64, n)
		for i := 0; i < n; i++ {
			if x2vData[i] > x1vData[i] {
				gxData[i] = gyData[i]
			}
		}
		gx := x1v.NewVec(float.SliceInterface(gxData))
		r.x2.AccGrad(gx)
	}

	return nil
}
