// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layernorm

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()

	// == Forward
	x := g.NewVariable(mat.NewVecDense([]float64{0.4, 0.8, -0.7, -0.5}), true)
	y := model.NewProc(g).Forward(x)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{1.15786, 0.2, -0.561559, -0.44465}, 1.0e-06) {
		t.Error("The output at position 0 doesn't match the expected values")
	}

	// == Backward
	y.PropagateGrad(mat.NewVecDense([]float64{-1.0, -0.2, 0.4, 0.6}))
	g.BackwardAll()

	if !floats.EqualApprox(x.Grad().Data(), []float64{-0.496258, 0.280667, -0.40876, 0.624352}, 1.0e-06) {
		t.Error("The x1-gradients don't match the expected values")
	}
	if !floats.EqualApprox(model.W.Grad().Data(), []float64{-0.64465, -0.25786, -0.451255, -0.483487}, 1.0e-06) {
		t.Error("The W-gradients don't match the expected values")
	}
	if !floats.EqualApprox(model.B.Grad().Data(), []float64{-1.0, -0.2, 0.4, 0.6}, 1.0e-06) {
		t.Error("The B-gradients don't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(4)
	model.W.Value().SetData([]float64{0.4, 0.0, -0.3, 0.8})
	model.B.Value().SetData([]float64{0.9, 0.2, -0.9, 0.2})
	return model
}
