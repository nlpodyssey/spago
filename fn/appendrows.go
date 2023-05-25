// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// AppendRows is a Function which appends new tail rows to a matrix.
type AppendRows[O DualValue] struct {
	x  O
	vs []O
}

// NewAppendRows returns a new AppendRows Function.
func NewAppendRows[O DualValue](x O, vs ...O) *AppendRows[O] {
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
func (a *AppendRows[O]) Forward() (mat.Matrix, error) {
	nodes := a.vs
	vs := make([]mat.Matrix, len(nodes))
	for i, n := range nodes {
		vs[i] = n.Value()
	}
	return a.x.Value().AppendRows(vs...), nil
}

// Backward computes the backward pass.
func (a *AppendRows[O]) Backward(gy mat.Matrix) error {
	xVal := a.x.Value()
	if gy.Rows() != xVal.Rows()+len(a.vs) {
		panic("fn: matrices have incompatible dimensions")
	}

	xRows := xVal.Rows()
	if a.x.RequiresGrad() {
		xGrads := gy.Slice(0, 0, xRows, xVal.Cols())
		a.x.AccGrad(xGrads)
	}

	for i, v := range a.vs {
		if !v.RequiresGrad() {
			continue
		}
		vGrads := gy.ExtractRow(xRows + i)
		v.AccGrad(vGrads)
	}
	return nil
}
