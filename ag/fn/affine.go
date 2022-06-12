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
	b         O
	w1        O
	x1        O
	wxPairs   []O
	operators []O // lazily computed
}

// NewAffine returns a new Affine Function.
//
// The affine transformation is in the form y = b + W1x1 + W2x2 + ... + WnXn.
// The function accepts the bias "b", the first mandatory pair of "w1" and "x1",
// and then an optional arbitrary list of (w, x) pairs, that is
// "w2", "x2", "w3", "x3", ..., "wN", "xN".
//
// For each additional pair (w, x), x is allowed to be nil. In this case, the
// pair is completely ignored.
// For example, given the arguments (b, w1, x1, w2, x2, w3, x3), if only x2
// is nil, the actual operation performed will be y = b + w1x1 + w3x3.
//
// If no additional (w, x) pair is given, or all x values of wxPairs are nil,
// the actual function is just y = b + W1x1.
//
// It panics if the length of wxs is not even.
func NewAffine[O Operand](b, w1, x1 O, wxPairs ...O) *Affine[O] {
	var filteredPairs []O
	if len(wxPairs) > 0 {
		if len(wxPairs)%2 != 0 {
			panic("mat: affine function: invalid list of additional (w, x) pairs")
		}
		filteredPairs = make([]O, 0, len(wxPairs))
		for i := 0; i < len(wxPairs); i += 2 {
			x := wxPairs[i+1]
			if operandIsNil(x) {
				continue
			}
			filteredPairs = append(filteredPairs, wxPairs[i], x)
		}
	}
	return &Affine[O]{
		b:       b,
		w1:      w1,
		x1:      x1,
		wxPairs: filteredPairs,
	}
}

// Operands returns the list of operands.
func (a *Affine[O]) Operands() []O {
	if a.operators == nil {
		ops := make([]O, len(a.wxPairs)+3)
		ops[0] = a.b
		ops[1] = a.w1
		ops[2] = a.x1
		copy(ops[3:], a.wxPairs)
		a.operators = ops
	}
	return a.operators
}

// Forward computes the output of the function.
func (a *Affine[O]) Forward() mat.Matrix {
	y := a.w1.Value().Mul(a.x1.Value()).AddInPlace(a.b.Value())

	wxPairs := a.wxPairs
	for i := 0; i < len(wxPairs); i += 2 {
		wx := wxPairs[i].Value().Mul(wxPairs[i+1].Value())
		y.AddInPlace(wx)
		mat.ReleaseMatrix(wx)
	}
	return y
}

// Backward computes the backward pass.
func (a *Affine[O]) Backward(gy mat.Matrix) {
	if a.b.RequiresGrad() {
		if !mat.SameDims(a.b.Value(), gy) {
			panic("fn: matrices have incompatible dimensions")
		}
		a.b.AccGrad(gy)
	}

	var wg sync.WaitGroup

	backwardWX := func(w, x O) {
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

	backwardWX(a.w1, a.x1)

	for i := 0; i < len(a.wxPairs); i += 2 {
		backwardWX(a.wxPairs[i], a.wxPairs[i+1])
	}

	wg.Wait()
}
