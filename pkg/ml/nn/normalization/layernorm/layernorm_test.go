// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layernorm

import (
	"github.com/saientist/spago/pkg/mat"
	"github.com/saientist/spago/pkg/ml/ag"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()

	// == Forward
	x := g.NewVariable(mat.NewVecDense([]float64{1.0, 2.0, 0.0, 4.0}), true)
	y := model.NewProc(g).Forward(x)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{0.1464537236, 0.2661938298, -0.154964787, 1.3170221268}, 1.0e-06) {
		t.Error("The output at position 0 doesn't match the expected values")
	}

	// == Backward
	y.PropagateGrad(mat.NewVecDense([]float64{-1.0, -0.2, 0.4, 0.6}))
	g.BackwardAll()

	if !floats.EqualApprox(x.Grad().Data(), []float64{-0.2889944606, -0.0208632365, 0.2271774637, 0.0826802334}, 1.0e-06) {
		t.Error("The x1-gradients don't match the expected values")
	}
	if !floats.EqualApprox(model.W.Grad().Data(), []float64{0.5070925528, -0.0338061702, -0.4732863826, 0.9127665951}, 1.0e-06) {
		t.Error("The W-gradients don't match the expected values")
	}
	if !floats.EqualApprox(model.B.Grad().Data(), []float64{-1.0, -0.2, 0.4, 0.6}, 1.0e-06) {
		t.Error("The B-gradients don't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(4)
	model.W.Value().SetData([]float64{0.5, -0.2, 0.3, 0.8})
	model.B.Value().SetData([]float64{0.4, 0.3, 0.2, 0.1})
	return model
}
