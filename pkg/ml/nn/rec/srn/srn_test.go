// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package srn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	y := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{-0.39693, -0.79688, 0.0, 0.70137, -0.18775}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{-1.32512, -0.55398, 1.0709, 0.5709}, 1.0e-05) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.W.Grad().Data(), []float64{
		0.65167, 0.73313, 0.73313, -0.81459,
		0.45167, 0.50813, 0.50813, -0.56459,
		-0.12, -0.135, -0.135, 0.15,
		0.38151, 0.4292, 0.4292, -0.47689,
		0.49221, 0.55374, 0.55374, -0.61527,
	}, 1.0e-05) {
		t.Error("W doesn't match the expected values")
	}

	if !floats.EqualApprox(model.B.Grad().Data(), []float64{
		-0.81459, -0.56459, 0.15, -0.47689, -0.61527,
	}, 1.0e-05) {
		t.Error("B doesn't match the expected values")
	}

	if model.WRec.HasGrad() {
		t.Error("WRec doesn't match the expected values")
	}
}

func TestModel_ForwardWithPrev(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()

	// == Forward
	x := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	yPrev := g.Tanh(g.NewVariable(mat.NewVecDense([]float64{-0.2, 0.2, -0.3, -0.9, -0.8}), true))
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	proc.SetInitialState(&State{Y: yPrev})
	y := proc.Forward(x)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{0.59539, -0.8115, 0.17565, 0.88075, 0.08444}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{-0.42553, -0.20751, 0.28232, 0.30119}, 0.005) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.W.Grad().Data(), []float64{
		-0.01311, -0.01475, -0.01475, 0.01639,
		0.42655, 0.47987, 0.47987, -0.53319,
		-0.25248, -0.28404, -0.28404, 0.3156,
		0.13623, 0.15326, 0.15326, -0.17029,
		0.29036, 0.32666, 0.32666, -0.36295,
	}, 1.0e-05) {
		t.Error("W doesn't match the expected values")
	}

	if !floats.EqualApprox(model.B.Grad().Data(), []float64{
		0.01639, -0.53319, 0.3156, -0.17029, -0.36295,
	}, 1.0e-05) {
		t.Error("B doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WRec.Grad().Data(), []float64{
		-0.00323, 0.00323, -0.00477, -0.01174, -0.01088,
		0.10524, -0.10524, 0.15533, 0.38193, 0.35406,
		-0.06229, 0.06229, -0.09194, -0.22606, -0.20957,
		0.03361, -0.03361, 0.04961, 0.12198, 0.11308,
		0.07164, -0.07164, 0.10573, 0.25998, 0.24101,
	}, 1.0e-05) {
		t.Error("WRec doesn't match the expected values")
	}
}

func newTestModel() *Model {
	params := New(4, 5)
	params.W.Value().SetData([]float64{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
		0.4, 1.0, -0.7, 0.8,
	})
	params.WRec.Value().SetData([]float64{
		0.0, 0.8, 0.8, -1.0, -0.7,
		-0.7, -0.8, 0.2, -0.7, 0.7,
		-0.9, 0.9, 0.7, -0.5, 0.5,
		0.0, -0.1, 0.5, -0.2, -0.8,
		-0.6, 0.6, 0.8, -0.1, -0.3,
	})
	params.B.Value().SetData([]float64{0.4, 0.0, -0.3, 0.8, -0.4})
	return params
}

func TestModel_ForwardSeq(t *testing.T) {
	model := newTestModel2()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	proc.SetInitialState(&State{
		Y: g.NewVariable(mat.NewVecDense([]float64{0.0, 0.0}), true),
	})

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{3.5, 4.0, -0.1}), true)
	_ = proc.Forward(x)
	s := proc.LastState()

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{-0.9732261643, 0.9987757968}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	x2 := g.NewVariable(mat.NewVecDense([]float64{3.3, -2.0, 0.1}), true)
	_ = proc.Forward(x2)
	s2 := proc.LastState()

	if !floats.EqualApprox(s2.Y.Value().Data(), []float64{-0.3773622668, 0.9671682519}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	s.Y.PropagateGrad(mat.NewVecDense([]float64{-0.0548928728, 0.4391429825}))
	s2.Y.PropagateGrad(mat.NewVecDense([]float64{-0.0527838886, 0.4222711092}))

	g.BackwardAll()

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.0015892366, 0.0013492153, -0.001893466}, 1.0e-05) {
		t.Error("The input gradients x don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{0.0308715656, 0.019034727, -0.0223609451}, 1.0e-05) {
		t.Error("The input gradients x2 don't match the expected values")
	}

	if !floats.EqualApprox(model.W.Grad().Data(), []float64{
		-0.1627089173, 0.0753041857, -0.0041459718,
		0.0936208886, -0.0504066848, 0.0026237982,
	}, 1.0e-05) {
		t.Error("W doesn't match the expected values")
	}

	if !floats.EqualApprox(model.B.Grad().Data(), []float64{
		-0.0490749674, 0.0283072608,
	}, 1.0e-05) {
		t.Error("B doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WRec.Grad().Data(), []float64{
		0.0440553621, -0.0452119261,
		-0.0265424287, 0.0272392341,
	}, 1.0e-05) {
		t.Error("WRec doesn't match the expected values")
	}
}

func newTestModel2() *Model {
	model := New(3, 2)
	model.W.Value().SetData([]float64{
		-0.2, -0.3, 0.5,
		0.8, 0.2, 0.01,
	})
	model.WRec.Value().SetData([]float64{
		0.5, 0.3,
		0.2, -0.1,
	})
	model.B.Value().SetData([]float64{-0.2, 0.1})
	return model
}
