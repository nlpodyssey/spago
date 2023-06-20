// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import "github.com/nlpodyssey/spago/mat"

// AppendRows is a Function which appends new tail rows to a matrix.
type AppendRows[O mat.Tensor] struct {
	x  O
	vs []O
}

// NewAppendRows returns a new AppendRows Function.
func NewAppendRows[O mat.Tensor](x O, vs ...O) *AppendRows[O] {
	return &AppendRows[O]{
		x:  x,
		vs: vs,
	}
}

// Operands returns the list of operands.
func (a *AppendRows[O]) Operands() []O {
	ops := make([]O, 0, len(a.vs)+1)
	ops = append(ops, a.x)
	ops = append(ops, a.vs...)
	return ops
}

// Forward computes the output of the function.
func (a *AppendRows[O]) Forward() (mat.Tensor, error) {
	nodes := a.vs
	vs := make([]mat.Matrix, len(nodes))
	for i, n := range nodes {
		vs[i] = n.Value().(mat.Matrix)
	}
	return a.x.Value().(mat.Matrix).AppendRows(vs...), nil
}

// Backward computes the backward pass.
func (a *AppendRows[O]) Backward(gy mat.Tensor) error {
	xVal := a.x.Value()
	if gy.Shape()[0] != xVal.Shape()[0]+len(a.vs) {
		panic("fn: matrices have incompatible dimensions")
	}

	xRows := xVal.Shape()[0]
	if a.x.RequiresGrad() {
		xGrads := gy.(mat.Matrix).Slice(0, 0, xRows, xVal.Shape()[1])
		a.x.AccGrad(xGrads)
	}

	for i, v := range a.vs {
		if !v.RequiresGrad() {
			continue
		}
		vGrads := gy.(mat.Matrix).ExtractRow(xRows + i)
		v.AccGrad(vGrads)
	}
	return nil
}
