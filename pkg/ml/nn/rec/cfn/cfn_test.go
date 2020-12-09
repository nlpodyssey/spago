// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfn

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

	if !floats.EqualApprox(y.Value().Data(), []float64{0.268, -0.025, 0.381, 0.613, -0.364}, 0.0005) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.318, 0.01, -0.027, 0.302}, 0.005) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WIn.Grad().Data(), []float64{
		0.039, 0.044, 0.044, -0.049,
		-0.012, -0.013, -0.013, 0.015,
		-0.081, -0.091, -0.091, 0.101,
		0.149, 0.167, 0.167, -0.186,
		-0.13, -0.146, -0.146, 0.162,
	}, 0.005) {
		t.Error("WIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BIn.Grad().Data(), []float64{-0.049, 0.015, 0.101, -0.186, 0.16}, 0.005) {
		t.Error("BIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		0.052, 0.059, 0.059, -0.065,
		0.154, 0.174, 0.174, -0.193,
		-0.089, -0.1, -0.1, 0.111,
		0.142, 0.159, 0.159, -0.177,
		0.104, 0.117, 0.117, -0.130,
	}, 0.005) {
		t.Error("WCand doesn't match the expected values")
	}

	if model.WInRec.HasGrad() {
		t.Error("WInRec doesn't match the expected values")
	}
	if model.WForRec.HasGrad() {
		t.Error("WForRec doesn't match the expected values")
	}

	if model.WFor.HasGrad() {
		t.Error("WFor doesn't match the expected values")
	}
}

func TestModel_ForwardWithPrev(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	proc.SetInitialState(
		&State{Y: g.NewVariable(mat.NewVecDense([]float64{-0.2, 0.2, -0.3, -0.9, -0.8}), true)},
	)

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	y := proc.Forward(x)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{0.308, 0.011, 0.405, 0.230, -0.689}, 0.0005) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.111, 0.37, -0.281, 0.126}, 0.005) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WIn.Grad().Data(), []float64{
		0.028, 0.032, 0.032, -0.035,
		-0.011, -0.012, -0.012, 0.014,
		-0.084, -0.094, -0.094, 0.105,
		0.144, 0.162, 0.162, -0.18,
		-0.182, -0.205, -0.205, 0.228,
	}, 0.005) {
		t.Error("WIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BIn.Grad().Data(), []float64{-0.035, 0.014, 0.105, -0.18, 0.228}, 0.005) {
		t.Error("BIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		0.082, 0.093, 0.093, -0.103,
		0.146, 0.164, 0.164, -0.183,
		-0.102, -0.115, -0.115, 0.128,
		0.226, 0.255, 0.255, -0.283,
		0.172, 0.194, 0.194, -0.215,
	}, 0.005) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WInRec.Grad().Data(), []float64{
		0.007, -0.007, 0.011, 0.032, 0.028,
		-0.003, 0.003, -0.004, -0.012, -0.011,
		-0.021, 0.021, -0.031, -0.094, -0.084,
		0.036, -0.036, 0.054, 0.162, 0.144,
		-0.046, 0.046, -0.068, -0.205, -0.182,
	}, 0.005) {
		t.Error("WInRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WFor.Grad().Data(), []float64{
		-0.003, -0.004, -0.004, 0.004,
		0.017, 0.019, 0.019, -0.022,
		0.006, 0.007, 0.007, -0.007,
		-0.177, -0.199, -0.199, 0.222,
		-0.144, -0.162, -0.162, 0.18,
	}, 0.005) {
		t.Error("WFor doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BFor.Grad().Data(), []float64{0.004, -0.022, -0.007, 0.222, 0.18}, 0.005) {
		t.Error("BFor doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WForRec.Grad().Data(), []float64{
		-0.001, 0.001, -0.001, -0.004, -0.003,
		0.004, -0.004, 0.006, 0.019, 0.017,
		0.001, -0.001, 0.002, 0.007, 0.006,
		-0.044, 0.044, -0.066, -0.199, -0.177,
		-0.036, 0.036, -0.054, -0.162, -0.144,
	}, 0.005) {
		t.Error("WForRec doesn't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(4, 5)
	model.WIn.Value().SetData([]float64{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
		0.4, 1.0, -0.7, 0.8,
	})
	model.WInRec.Value().SetData([]float64{
		0.0, 0.8, 0.8, -1.0, -0.7,
		-0.7, -0.8, 0.2, -0.7, 0.7,
		-0.9, 0.9, 0.7, -0.5, 0.5,
		0.0, -0.1, 0.5, -0.2, -0.8,
		-0.6, 0.6, 0.8, -0.1, -0.3,
	})
	model.BIn.Value().SetData([]float64{0.4, 0.0, -0.3, 0.8, -0.4})
	model.WFor.Value().SetData([]float64{
		0.1, 0.4, -1.0, 0.4,
		0.7, -0.2, 0.1, 0.0,
		0.7, 0.8, -0.5, -0.3,
		-0.9, 0.9, -0.3, -0.3,
		-0.7, 0.6, -0.6, -0.8,
	})
	model.WForRec.Value().SetData([]float64{
		0.1, -0.6, -1.0, -0.1, -0.4,
		0.5, -0.9, 0.0, 0.8, 0.3,
		-0.3, -0.9, 0.3, 1.0, -0.2,
		0.7, 0.2, 0.3, -0.4, -0.6,
		-0.2, 0.5, -0.2, -0.9, 0.4,
	})
	model.BFor.Value().SetData([]float64{0.9, 0.2, -0.9, 0.2, -0.9})
	model.WCand.Value().SetData([]float64{
		-1.0, 0.2, 0.0, 0.2,
		-0.7, 0.7, -0.3, -0.3,
		0.3, -0.6, 0.0, 0.7,
		-1.0, -0.6, 0.9, 0.8,
		0.5, 0.8, -0.9, -0.8,
	})
	return model
}

func TestModel_ForwardSeq(t *testing.T) {
	model := newTestModel2()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	proc.SetInitialState(
		&State{Y: g.NewVariable(mat.NewVecDense([]float64{0.0, 0.0}), true)},
	)

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{3.5, 4.0, -0.1}), true)
	_ = proc.Forward(x)
	s := proc.LastState()

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{-0.0886045623, 0.9749300057}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	x2 := g.NewVariable(mat.NewVecDense([]float64{3.3, -2.0, 0.1}), true)
	_ = proc.Forward(x2)
	s2 := proc.LastState()

	if !floats.EqualApprox(s2.Y.Value().Data(), []float64{0.2205790544, 0.5834192006}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	s.Y.PropagateGrad(mat.NewVecDense([]float64{-0.0522186536, 0.4177492291}))
	s2.Y.PropagateGrad(mat.NewVecDense([]float64{-0.0436513876, 0.3492111007}))

	g.BackwardAll()

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.0087725508, 0.0021613524, 0.000922185}, 1.0e-05) {
		t.Error("The input gradients x don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{0.1519336751, 0.2004414748, -0.1394754602}, 1.0e-05) {
		t.Error("The input gradients x2 don't match the expected values")
	}

	if !floats.EqualApprox(model.WFor.Grad().Data(), []float64{
		0.0021317013, -0.0012919402, 0.000064597,
		0.1915774162, -0.116107525, 0.0058053762,
	}, 1.0e-05) {
		t.Error("WFor doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BFor.Grad().Data(), []float64{
		0.0006459701, 0.0580537625,
	}, 1.0e-05) {
		t.Error("BFor doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WIn.Grad().Data(), []float64{
		-0.0086247377, 0.0259952369, -0.0009604806,
		0.0488884721, 0.0399603087, -0.0008611542,
	}, 1.0e-05) {
		t.Error("WIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BIn.Grad().Data(), []float64{
		-0.0028191819, 0.0141256817,
	}, 1.0e-05) {
		t.Error("BIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		-0.053950159, 0.0250862936, -0.0013786491,
		1.0348122585, -0.62172667, 0.0311750786,
	}, 1.0e-05) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WForRec.Grad().Data(), []float64{
		-5.72358971272346e-005, 0.0006297756,
		-0.0051438282, 0.056598355,
	}, 1.0e-05) {
		t.Error("WForRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WInRec.Grad().Data(), []float64{
		0.000550411, -0.0060562594,
		-0.000244289, 0.00268795,
	}, 1.0e-05) {
		t.Error("WInRec doesn't match the expected values")
	}
}

func newTestModel2() *Model {
	model := New(3, 2)
	model.WIn.Value().SetData([]float64{
		-0.2, -0.3, 0.5,
		0.8, 0.2, 0.01,
	})
	model.WInRec.Value().SetData([]float64{
		0.5, 0.3,
		0.2, -0.1,
	})
	model.BIn.Value().SetData([]float64{-0.2, 0.1})
	model.WFor.Value().SetData([]float64{
		0.3, 0.2, -0.4,
		0.4, 0.1, -0.6,
	})
	model.WForRec.Value().SetData([]float64{
		-0.5, 0.22,
		0.8, -0.6,
	})
	model.BFor.Value().SetData([]float64{0.5, 0.3})
	model.WCand.Value().SetData([]float64{
		-0.001, -0.3, 0.5,
		0.4, 0.6, -0.3,
	})
	return model
}
