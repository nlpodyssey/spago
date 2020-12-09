// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gru

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

	if !floats.EqualApprox(y.Value().Data(), []float64{0.74, -0.23, 0.11, 0.49, -0.05}, 0.005) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.53, -0.49, 0.18, 0.20}, 0.005) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WPart.Grad().Data(), []float64{
		-0.01, -0.02, -0.02, 0.02,
		-0.10, -0.12, -0.12, 0.13,
		-0.02, -0.02, -0.02, 0.03,
		0.22, 0.24, 0.24, -0.27,
		-0.02, -0.02, -0.02, 0.02,
	}, 0.005) {
		t.Error("WPart doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BPart.Grad().Data(), []float64{0.02, 0.13, 0.03, -0.27, 0.02}, 0.005) {
		t.Error("BPart doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		-0.03, -0.03, -0.03, 0.04,
		0.24, 0.27, 0.27, -0.30,
		0.00, 0.00, 0.00, 0.00,
		0.06, 0.06, 0.06, -0.07,
		0.09, 0.10, 0.10, -0.12,
	}, 0.005) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BCand.Grad().Data(), []float64{0.04, -0.3, 0.0, -0.07, -0.12}, 0.005) {
		t.Error("BCand doesn't match the expected values")
	}

	if model.BRes.HasGrad() {
		t.Error("BRes doesn't match the expected values")
	}

	if model.WRes.HasGrad() {
		t.Error("WRes doesn't match the expected values")
	}

	if model.WPartRec.HasGrad() {
		t.Error("WPartRec doesn't match the expected values")
	}

	if model.WResRec.HasGrad() {
		t.Error("WResRec doesn't match the expected values")
	}

	if model.WCandRec.HasGrad() {
		t.Error("WCandRec doesn't match the expected values")
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

	if !floats.EqualApprox(y.Value().Data(), []float64{0.86, 0.18, -0.24, 0.36, -0.36}, 0.005) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.56, -0.83, 0.5, 0.55}, 0.005) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WPart.Grad().Data(), []float64{
		-0.02, -0.02, -0.02, 0.03,
		-0.01, -0.01, -0.01, 0.01,
		0.0, 0.0, 0.0, -0.01,
		0.42, 0.47, 0.47, -0.52,
		0.17, 0.2, 0.2, -0.22,
	}, 0.005) {
		t.Error("WPart doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BPart.Grad().Data(), []float64{0.03, 0.01, -0.01, -0.52, -0.22}, 0.005) {
		t.Error("BPart doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		-0.02, -0.02, -0.02, 0.02,
		0.08, 0.09, 0.09, -0.10,
		0.00, 0.00, 0.00, 0.00,
		0.05, 0.05, 0.05, -0.06,
		0.22, 0.25, 0.25, -0.28,
	}, 0.005) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BCand.Grad().Data(), []float64{0.02, -0.1, 0.0, -0.06, -0.28}, 0.005) {
		t.Error("BCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WRes.Grad().Data(), []float64{
		0.0, 0.01, 0.01, -0.01,
		0.0, 0.0, 0.0, 0.0,
		0.01, 0.01, 0.01, -0.02,
		0.02, 0.02, 0.02, -0.03,
		0.01, 0.01, 0.01, -0.01,
	}, 0.005) {
		t.Error("WRes doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BRes.Grad().Data(), []float64{-0.01, 0.0, -0.02, -0.03, -0.01}, 0.005) {
		t.Error("BRes doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WPartRec.Grad().Data(), []float64{
		-0.01, 0.01, -0.01, -0.02, -0.02,
		0.0, 0.0, 0.0, -0.01, -0.01,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, -0.1, 0.16, 0.47, 0.42,
		0.04, -0.04, 0.07, 0.2, 0.17,
	}, 0.005) {
		t.Error("WPartRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WResRec.Grad().Data(), []float64{
		0.0, 0.0, 0.0, 0.01, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.01, 0.01,
		0.01, -0.01, 0.01, 0.02, 0.02,
		0.0, 0.0, 0.0, 0.01, 0.01,
	}, 0.005) {
		t.Error("WResRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCandRec.Grad().Data(), []float64{
		0.0, 0.0, 0.0, -0.01, -0.01,
		0.01, -0.01, 0.02, 0.08, 0.04,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.01, 0.0, 0.01, 0.04, 0.02,
		0.04, -0.01, 0.05, 0.21, 0.12,
	}, 0.005) {
		t.Error("WCandRec doesn't match the expected values")
	}
}

func newTestModel() *Model {

	params := New(4, 5)

	params.WPart.Value().SetData([]float64{
		0.1, 0.4, -1.0, 0.4,
		0.7, -0.2, 0.1, 0.0,
		0.7, 0.8, -0.5, -0.3,
		-0.9, 0.9, -0.3, -0.3,
		-0.7, 0.6, -0.6, -0.8,
	})

	params.WPartRec.Value().SetData([]float64{
		0.1, -0.6, -1.0, -0.1, -0.4,
		0.5, -0.9, 0.0, 0.8, 0.3,
		-0.3, -0.9, 0.3, 1.0, -0.2,
		0.7, 0.2, 0.3, -0.4, -0.6,
		-0.2, 0.5, -0.2, -0.9, 0.4,
	})

	params.BPart.Value().SetData([]float64{0.9, 0.2, -0.9, 0.2, -0.9})

	params.WRes.Value().SetData([]float64{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
		0.4, 1.0, -0.7, 0.8,
	})

	params.WResRec.Value().SetData([]float64{
		0.0, 0.8, 0.8, -1.0, -0.7,
		-0.7, -0.8, 0.2, -0.7, 0.7,
		-0.9, 0.9, 0.7, -0.5, 0.5,
		0.0, -0.1, 0.5, -0.2, -0.8,
		-0.6, 0.6, 0.8, -0.1, -0.3,
	})

	params.BRes.Value().SetData([]float64{0.4, 0.0, -0.3, 0.8, -0.4})

	params.WCand.Value().SetData([]float64{
		-1.0, 0.2, 0.0, 0.2,
		-0.7, 0.7, -0.3, -0.3,
		0.3, -0.6, 0.0, 0.7,
		-1.0, -0.6, 0.9, 0.8,
		0.5, 0.8, -0.9, -0.8,
	})

	params.WCandRec.Value().SetData([]float64{
		0.2, -0.3, -0.3, -0.5, -0.7,
		0.4, -0.1, -0.6, -0.4, -0.8,
		0.6, 0.6, 0.1, 0.7, -0.4,
		-0.8, 0.9, 0.1, -0.1, -0.2,
		-0.5, -0.3, -0.6, -0.6, 0.1,
	})

	params.BCand.Value().SetData([]float64{0.5, -0.5, 1.0, 0.4, 0.9})

	return params
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

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{-0.634733134450701, 0.896135841414256}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	x2 := g.NewVariable(mat.NewVecDense([]float64{3.3, -2.0, 0.1}), true)
	_ = proc.Forward(x2)
	s2 := proc.LastState()

	if !floats.EqualApprox(s2.Y.Value().Data(), []float64{0.646126994447876, 0.537141024639326}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	s.Y.PropagateGrad(mat.NewVecDense([]float64{-0.052008468343874, 0.416067746750988}))
	s2.Y.PropagateGrad(mat.NewVecDense([]float64{-0.041704888674704, 0.333639109397627}))

	g.BackwardAll()

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.022626682234541, 0.019282896989004, -0.05477940973827}, 1.0e-05) {
		t.Error("The input gradients x don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{0.047347465801696, 0.102160284950441, -0.023609485283631}, 1.0e-05) {
		t.Error("The input gradients x2 don't match the expected values")
	}

	if !floats.EqualApprox(model.WPart.Grad().Data(), []float64{
		-0.013817689446019, 0.037543919616777, -0.001400663526169,
		0.016466177962923, 0.310541096478206, -0.010290827228346,
	}, 1.0e-05) {
		t.Error("WPart doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BPart.Grad().Data(), []float64{
		-0.004475986168291, 0.001816279627817,
	}, 1.0e-05) {
		t.Error("BPart doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WRes.Grad().Data(), []float64{
		-0.008595802515716, 0.005209577282252, -0.000260478864113,
		-0.006145984396192, 0.003724839027995, -0.0001862419514,
	}, 1.0e-05) {
		t.Error("WRes doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BRes.Grad().Data(), []float64{
		-0.002604788641126, -0.001862419513998,
	}, 1.0e-05) {
		t.Error("BRes doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		-0.167811472603026, -0.159044185005926, 0.003692462934823,
		0.590071690390826, -0.35441204273124, 0.017772996392842,
	}, 1.0e-05) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BCand.Grad().Data(), []float64{
		-0.048270296961236, 0.17877784905402,
	}, 1.0e-05) {
		t.Error("BCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WResRec.Grad().Data(), []float64{
		0.001653345658763, -0.002334244460622,
		0.001182139375782, -0.001668980878242,
	}, 1.0e-05) {
		t.Error("WResRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WPartRec.Grad().Data(), []float64{
		0.005865766116558, -0.008281469753348,
		0.032083218683093, -0.045296078949356,
	}, 1.0e-05) {
		t.Error("WPartRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCandRec.Grad().Data(), []float64{
		0.001568934580973, -0.004540982737526,
		-0.049299649456851, 0.142688458693303,
	}, 1.0e-05) {
		t.Error("WCandRec doesn't match the expected values")
	}
}

func newTestModel2() *Model {
	model := New(3, 2)
	model.WRes.Value().SetData([]float64{
		-0.2, -0.3, 0.5,
		0.8, 0.2, 0.01,
	})
	model.WResRec.Value().SetData([]float64{
		0.5, 0.3,
		0.2, -0.1,
	})
	model.BRes.Value().SetData([]float64{-0.2, 0.1})
	model.WPart.Value().SetData([]float64{
		0.3, 0.2, -0.4,
		0.4, 0.1, -0.6,
	})
	model.WPartRec.Value().SetData([]float64{
		-0.5, 0.22,
		0.8, -0.6,
	})
	model.BPart.Value().SetData([]float64{0.5, 0.3})
	model.WCand.Value().SetData([]float64{
		-0.001, -0.3, 0.5,
		0.4, 0.6, -0.3,
	})
	model.WCandRec.Value().SetData([]float64{
		0.2, 0.7,
		0.1, -0.1,
	})
	model.BCand.Value().SetData([]float64{0.4, 0.3})
	return model
}
