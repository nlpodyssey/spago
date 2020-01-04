// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package highway

import (
	"brillion.io/spago/pkg/mat"
	"brillion.io/spago/pkg/ml/act"
	"brillion.io/spago/pkg/ml/ag"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_Forward(t *testing.T) {

	model := newTestModel()
	g := ag.NewGraph()

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	y := model.NewProc(g).Forward(x)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{-0.456097, -0.855358, -0.79552, 0.844718}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	g.Backward(y, mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64}))

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.822396, 0.132595, -0.437002, 0.446894}, 1.0e-06) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WIn.Grad().(*mat.Dense).Data(), []float64{
		-0.327765, -0.368736, -0.368736, 0.409706,
		-0.094803, -0.106653, -0.106653, 0.118504,
		0.013931, 0.015672, 0.015672, -0.017413,
		-0.346622, -0.389949, -0.389949, 0.433277,
	}, 1.0e-06) {
		t.Error("WIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BIn.Grad().Data(), []float64{
		0.409706, 0.118504, -0.017413, 0.433277,
	}, 1.0e-06) {
		t.Error("BIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WT.Grad().(*mat.Dense).Data(), []float64{
		-0.023020, -0.025897, -0.025897, 0.028775,
		-0.015190, -0.017088, -0.017088, 0.018987,
		0.011082, 0.012467, 0.012467, -0.013853,
		0.097793, 0.110017, 0.110017, -0.122241,
	}, 1.0e-06) {
		t.Error("WT doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BT.Grad().Data(), []float64{
		0.028775, 0.018987, -0.013853, -0.122241,
	}, 1.0e-06) {
		t.Error("BT doesn't match the expected values")
	}
}

func newTestModel() *Model {

	model := New(4, act.Tanh)

	model.WIn.Value().SetData([]float64{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
	})

	model.WT.Value().SetData([]float64{
		0.1, 0.4, -1.0, 0.4,
		0.7, -0.2, 0.1, 0.0,
		0.7, 0.8, -0.5, -0.3,
		-0.9, 0.9, -0.3, -0.3,
	})

	model.BIn.Value().SetData([]float64{0.4, 0.0, -0.3, 0.8})
	model.BT.Value().SetData([]float64{0.9, 0.2, -0.9, 0.2})

	return model
}
