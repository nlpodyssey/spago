// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"
	"reflect"
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

// Affine is an operator to apply the affine function y = b + W1x1 + W2x2 + ... + WnXn.
type Affine[O mat.Tensor] struct {
	b       O
	w1      O
	x1      O
	wxPairs []O
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
func NewAffine[O mat.Tensor](b, w1, x1 O, wxPairs ...O) *Affine[O] {
	var filteredPairs []O
	if len(wxPairs) > 0 {
		if len(wxPairs)%2 != 0 {
			panic("mat: affine function: invalid list of additional (w, x) pairs")
		}
		filteredPairs = make([]O, 0, len(wxPairs))
		for i := 0; i < len(wxPairs); i += 2 {
			x := wxPairs[i+1]
			if isNil(x) {
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
	ops := make([]O, 0, len(a.wxPairs)+3)
	ops = append(ops, a.b, a.w1, a.x1)
	ops = append(ops, a.wxPairs...)
	return ops
}

// Forward computes the output of the function.
func (a *Affine[O]) Forward() (mat.Tensor, error) {
	y := a.w1.Value().(mat.Matrix).Mul(a.x1.Value().(mat.Matrix)).AddInPlace(a.b.Value().(mat.Matrix))

	wxPairs := a.wxPairs
	for i := 0; i < len(wxPairs); i += 2 {
		wx := wxPairs[i].Value().(mat.Matrix).Mul(wxPairs[i+1].Value().(mat.Matrix))
		y.AddInPlace(wx)
	}
	return y, nil
}

// Backward computes the backward pass.
func (a *Affine[O]) Backward(gy mat.Tensor) error {
	if a.b.RequiresGrad() {
		if !mat.SameDims(a.b.Value().(mat.Matrix), gy.(mat.Matrix)) {
			return fmt.Errorf("fn: matrices have incompatible dimensions")
		}
		a.b.AccGrad(gy)
	}

	var wg sync.WaitGroup

	backwardWX := func(w, x O) {
		wv := w.Value().(mat.Matrix)
		xv := x.Value().(mat.Matrix)

		if wv.Shape()[0] != gy.Shape()[0] || xv.Shape()[1] != gy.Shape()[1] {
			panic("fn: matrices have incompatible dimensions")
		}

		if w.RequiresGrad() {
			wg.Add(1)
			go func() {
				xt := xv.T()
				gx := gy.(mat.Matrix).Mul(xt)
				w.AccGrad(gx)
				wg.Done()
			}()
		}

		if x.RequiresGrad() {
			wg.Add(1)
			go func() {
				if gy.Shape()[1] == 1 {
					gx := wv.MulT(gy.(mat.Matrix))
					x.AccGrad(gx)
				} else {
					wt := wv.T()
					gx := wt.Mul(gy.(mat.Matrix))
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
	return nil
}

func isNil[O mat.Tensor](o O) bool {
	if any(o) == nil {
		return true
	}
	v := reflect.ValueOf(o)
	return v.Kind() == reflect.Pointer && v.IsNil()
}
