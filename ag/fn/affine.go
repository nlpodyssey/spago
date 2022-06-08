// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

// Affine is an operator to apply the affine function y = b + W1x1 + W2x2 + ... + WnXn.
type Affine[O Operand] struct {
	operands []O
}

// NewAffine returns a new Affine Function.
//
// The affine transformation is in the form y = b + W1x1 + W2x2 + ... + WnXn.
// The function accepts the bias "b" and an arbitrary list of (w, x) pairs:
// (b, w1, x1, w2, x2, ..., wn, xn).
//
// For each given pair (w, x), x is allowed to be nil. In this case, the pair
// is completely ignored.
// For example, given the arguments (b, w1, x1, w2, x2, w3, x3), if only x2
// is nil, the actual operation performed will be y = b + w1x1 + w3x3.
//
// If no (w, x) pair is given, or all x values are nil, the actual function
// is just y = b
//
// It panics if the length of wxs is not even.
func NewAffine[O Operand](b O, wxs ...O) *Affine[O] {
	if len(wxs)%2 != 0 {
		panic("mat: affine function: invalid list of (w, x) pairs")
	}
	operands := make([]O, 1, len(wxs)+1)
	operands[0] = b
	for i := 0; i < len(wxs); i += 2 {
		x := wxs[i+1]
		if operandIsNil(x) {
			continue
		}
		operands = append(operands, wxs[i])
		operands = append(operands, x)
	}
	return &Affine[O]{
		operands: operands,
	}
}

// Operands returns the list of operands.
func (a *Affine[O]) Operands() []O {
	return a.operands
}

// Forward computes the output of the function.
func (a *Affine[O]) Forward() mat.Matrix {
	operands := a.operands
	y := operands[0].Value().Clone()
	for i := 1; i < len(operands); i += 2 {
		w := operands[i].Value()
		x := operands[i+1].Value()
		wx := w.Mul(x)
		y.AddInPlace(wx)
		mat.ReleaseMatrix(wx)
	}
	return y
}

// Backward computes the backward pass.
func (a *Affine[O]) Backward(gy mat.Matrix) {
	operands := a.operands

	if b := operands[0]; b.RequiresGrad() {
		if !mat.SameDims(b.Value(), gy) {
			panic("fn: matrices have incompatible dimensions")
		}
		b.AccGrad(gy)
	}

	var wg sync.WaitGroup

	for i := 1; i < len(operands); i += 2 {
		w := operands[i]
		x := operands[i+1]

		wv := w.Value()
		xv := x.Value()

		if wv.Rows() != gy.Rows() || xv.Columns() != gy.Columns() {
			panic("fn: matrices have incompatible dimensions")
		}

		if w.RequiresGrad() {
			wg.Add(1)
			go func() {
				xt := xv.T()
				defer mat.ReleaseMatrix(xt)
				gx := gy.Mul(xt)
				defer mat.ReleaseMatrix(gx)
				w.AccGrad(gx)
				wg.Done()
			}()
		}

		if x.RequiresGrad() {
			wg.Add(1)
			go func() {
				if gy.Columns() == 1 {
					gx := wv.MulT(gy)
					defer mat.ReleaseMatrix(gx)
					x.AccGrad(gx)
				} else {
					wt := wv.T()
					defer mat.ReleaseMatrix(wt)
					gx := wt.Mul(gy)
					defer mat.ReleaseMatrix(gx)
					x.AccGrad(gx)
				}
				wg.Done()
			}()
		}
	}

	wg.Wait()
}
