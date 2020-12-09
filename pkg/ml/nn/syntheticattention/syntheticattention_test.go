// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntheticattention

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_SyntheticAttention(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training})

	x1 := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{0.8, -0.3, 0.5, 0.3}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{-0.2, 0.7, 0.2, 0.4}), true)

	// == Forward
	output := proc.Forward(x1, x2, x3)

	if !floats.EqualApprox(output[0].Value().Data(), []float64{0.778722, -0.5870107, -0.36687185}, 1.0e-05) {
		t.Error("The output[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(output[1].Value().Data(), []float64{0.8265906, -0.7254198, -0.3560310}, 1.0e-05) {
		t.Error("The output[1] doesn't match the expected values")
	}
	if !floats.EqualApprox(output[2].Value().Data(), []float64{0.7651726, -0.5680948, -0.3797024}, 1.0e-05) {
		t.Error("The output[2] doesn't match the expected values")
	}

	// == Backward
	output[0].PropagateGrad(mat.NewVecDense([]float64{-0.04, 0.36, 0.32}))
	output[1].PropagateGrad(mat.NewVecDense([]float64{-0.08, -0.2, -0.1}))
	output[2].PropagateGrad(mat.NewVecDense([]float64{0.1, 0.3, 0.8}))

	g.BackwardAll()

	if !floats.EqualApprox(x1.Grad().Data(), []float64{-0.0015012, 0.40111136, -0.26565675, -0.1677756}, 1.0e-05) {
		t.Error("The input gradients x1 don't match the expected values")
	}
	if !floats.EqualApprox(x2.Grad().Data(), []float64{0.0472872, 0.19752994, -0.14673837, -0.0747287}, 1.0e-05) {
		t.Error("The input gradients x2 don't match the expected values")
	}
	if !floats.EqualApprox(x3.Grad().Data(), []float64{0.02086887, 0.2511595, -0.1880584, -0.07230184}, 1.0e-05) {
		t.Error("The input gradients x3 don't match the expected values")
	}
	if !floats.EqualApprox(model.Value.W.Grad().(*mat.Dense).Data(), []float64{
		0.02565823, 0.01689616, 0.02447476, -0.02306554,
		-0.0871563, -0.1083235, -0.0844748, 0.287951612,
		-0.2523685, -0.2838154, -0.2480056, 0.669627458,
	}, 1.0e-05) {
		t.Error("Value W doesn't match the expected values")
	}
	if !floats.EqualApprox(model.Value.B.Grad().Data(), []float64{
		-0.02, 0.46, 1.02,
	}, 1.0e-05) {
		t.Error("Value B doesn't match the expected values")
	}
	if !floats.EqualApprox(model.FFN.Layers[0].(*linear.Model).W.Grad().Data(), []float64{
		0.10050248, 0.03293678, 0.08198445, -0.0819892,
		0.0, 0.0, 0.0, 0.0,
		-0.0075098, 0.0028162, -0.0046936, -0.00281620,
		0.00989979, -0.034649, -0.0098997, -0.01979958,
	}, 1.0e-05) {
		t.Error("MLP input W doesn't match the expected values")
	}
	if !floats.EqualApprox(model.FFN.Layers[0].(*linear.Model).B.Grad().Data(), []float64{
		-0.0779462, 0.0, -0.0093873, -0.04949895,
	}, 1.0e-05) {
		t.Error("FFN input B doesn't match the expected values")
	}
	if !floats.EqualApprox(model.W.Grad().Data(), []float64{
		-0.0589030, 0.0, 0.02524230, -0.02234341,
		-0.0300720, 0.0, -0.0014955, -0.01170433,
		0.08897514, 0.0, -0.0237467, 0.034047752,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
	}, 1.0e-05) {
		t.Error("W doesn't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(Config{
		InputSize:  4,
		HiddenSize: 4,
		MaxLength:  5,
		ValueSize:  3,
	})
	model.Value.W.Value().SetData([]float64{
		0.5, 0.6, -0.8, 0.7,
		-0.4, 0.1, 0.7, -0.7,
		0.3, 0.8, -0.9, 0.0,
	})
	model.Value.B.Value().SetData([]float64{0.4, 0.0, -0.3})
	model.FFN.Layers[0].(*linear.Model).W.Value().SetData([]float64{
		0.7, -0.2, -0.1, 0.2,
		-0.1, -0.1, 0.3, -0.2,
		0.6, 0.1, 0.9, 0.3,
		-0.3, 0.3, 0.4, -0.8,
	})
	model.FFN.Layers[0].(*linear.Model).B.Value().SetData([]float64{0.8, -0.2, -0.5, 0.2})
	model.W.Value().SetData([]float64{
		0.4, 0.3, 0.2, -0.5,
		-0.9, -0.4, 0.1, -0.4,
		-0.3, 0.3, 0.4, -0.8,
		0.7, 0.8, 0.2, 0.3,
		0.5, 0.2, 0.4, -0.8,
	})
	return model
}
