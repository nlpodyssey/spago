// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// AppendRows is a Function which appends new tail rows to a matrix.
type AppendRows[O Operand] struct {
	x        O
	vs       []O
	operands []O
}

// NewAppendRows returns a new AppendRows Function.
func NewAppendRows[O Operand](x O, vs ...O) *AppendRows[O] {
	operands := make([]O, len(vs)+1)
	operands[0] = x
	copy(operands[1:], vs)

	return &AppendRows[O]{
		x:        x,
		vs:       vs,
		operands: operands,
	}
}

// Operands returns the list of operands.
func (a *AppendRows[O]) Operands() []O {
	return a.operands
}

// Forward computes the output of the function.
func (a *AppendRows[O]) Forward() mat.Matrix {
	nodes := a.vs
	vs := make([]mat.Matrix, len(nodes))
	for i, n := range nodes {
		vs[i] = n.Value()
	}
	return a.x.Value().AppendRows(vs...)
}

// Backward computes the backward pass.
func (a *AppendRows[O]) Backward(gy mat.Matrix) {
	xVal := a.x.Value()
	if gy.Rows() != xVal.Rows()+len(a.vs) {
		panic("fn: matrices have incompatible dimensions")
	}

	xRows := xVal.Rows()
	if a.x.RequiresGrad() {
		xGrads := gy.Slice(0, 0, xRows, xVal.Columns())
		a.x.AccGrad(xGrads)
		mat.ReleaseMatrix(xGrads)
	}

	for i, v := range a.vs {
		if !v.RequiresGrad() {
			continue
		}
		vGrads := gy.ExtractRow(xRows + i)
		v.AccGrad(vGrads)
		mat.ReleaseMatrix(vGrads)
	}
}
