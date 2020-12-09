// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ran

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
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training})

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	_ = proc.Forward(x)[0]
	s := proc.(*Processor).LastState()

	if !floats.EqualApprox(s.InG.Value().Data(), []float64{0.39652, 0.25162, 0.5, 0.70475, 0.45264}, 1.0e-05) {
		t.Error("The inG doesn't match the expected values")
	}

	if !floats.EqualApprox(s.ForG.Value().Data(), []float64{0.85321, 0.43291, 0.11609, 0.51999, 0.24232}, 1.0e-05) {
		t.Error("The forG doesn't match the expected values")
	}

	if !floats.EqualApprox(s.Cand.Value().Data(), []float64{1.02, -0.1, 0.1, 2.03, -1.41}, 1.0e-05) {
		t.Error("The candidate doesn't match the expected values")
	}

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{0.38375, -0.02516, 0.04996, 0.8918, -0.56369}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, s.Y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.21996, -0.12731, 0.10792, 0.49361}, 1.0e-05) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WIn.Grad().Data(), []float64{
		0.03101, 0.03489, 0.03489, -0.03877,
		-0.01167, -0.01313, -0.01313, 0.01459,
		-0.00399, -0.00449, -0.00449, 0.00499,
		0.05175, 0.05822, 0.05822, -0.06469,
		-0.19328, -0.21744, -0.21744, 0.2416,
	}, 1.0e-05) {
		t.Error("WIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BIn.Grad().Data(), []float64{-0.03877, 0.01459, 0.00499, -0.06469, 0.2416}, 1.0e-05) {
		t.Error("BIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		0.05038, 0.05668, 0.05668, -0.06298,
		0.15594, 0.17543, 0.17543, -0.19492,
		-0.07978, -0.08976, -0.08976, 0.09973,
		0.08635, 0.09714, 0.09714, -0.10794,
		0.25044, 0.28174, 0.28174, -0.31304,
	}, 1.0e-05) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BCand.Grad().Data(), []float64{-0.06298, -0.19492, 0.09973, -0.10794, -0.31304}, 1.0e-05) {
		t.Error("BCand doesn't match the expected values")
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
	proc.SetInitialState(&State{
		C: g.NewVariable(mat.NewVecDense([]float64{-0.2, 0.2, -0.3, -0.9, -0.8}), true),
		Y: g.NewVariable(mat.NewVecDense([]float64{-0.2, 0.2, -0.3, -0.9, -0.8}), true),
	})

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	_ = proc.Forward(x)
	s := proc.LastState()

	if !floats.EqualApprox(s.InG.Value().Data(), []float64{0.72312, 0.24974, 0.54983, 0.82054, 0.53494}, 1.0e-05) {
		t.Error("The inG doesn't match the expected values")
	}

	if !floats.EqualApprox(s.ForG.Value().Data(), []float64{0.91133, 0.18094, 0.04834, 0.67481, 0.38936}, 1.0e-05) {
		t.Error("The forG doesn't match the expected values")
	}

	if !floats.EqualApprox(s.Cand.Value().Data(), []float64{1.02, -0.1, 0.1, 2.03, -1.41}, 1.0e-05) {
		t.Error("The candidate doesn't match the expected values")
	}

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{0.5045, 0.01121, 0.04046, 0.78504, -0.78786}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, s.Y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.19694, 0.11428, -0.14008, 0.15609}, 1.0e-05) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WIn.Grad().Data(), []float64{
		0.00798, 0.00898, 0.00898, -0.00997,
		-0.01107, -0.01246, -0.01246, 0.01384,
		-0.00377, -0.00424, -0.00424, 0.00471,
		0.07845, 0.08826, 0.08826, -0.09807,
		-0.13175, -0.14822, -0.14822, 0.16469,
	}, 1.0e-05) {
		t.Error("WIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BIn.Grad().Data(), []float64{-0.00997, 0.01384, 0.00471, -0.09807, 0.16469}, 1.0e-05) {
		t.Error("BIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		0.02825, 0.03178, 0.03178, -0.03531,
		0.14759, 0.16603, 0.16603, -0.18448,
		-0.08364, -0.09409, -0.09409, 0.10455,
		0.21535, 0.24227, 0.24227, -0.26919,
		0.20092, 0.22604, 0.22604, -0.25115,
	}, 1.0e-05) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BCand.Grad().Data(), []float64{-0.03531, -0.18448, 0.10455, -0.26919, -0.25115}, 1.0e-05) {
		t.Error("BCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WInRec.Grad().Data(), []float64{
		0.00199, -0.00199, 0.00299, 0.00898, 0.00798,
		-0.00277, 0.00277, -0.00415, -0.01246, -0.01107,
		-0.00094, 0.00094, -0.00141, -0.00424, -0.00377,
		0.01961, -0.01961, 0.02942, 0.08826, 0.07845,
		-0.03294, 0.03294, -0.04941, -0.14822, -0.13175,
	}, 1.0e-05) {
		t.Error("WInRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WForRec.Grad().Data(), []float64{
		-0.00016, 0.00016, -0.00024, -0.00071, -0.00063,
		0.00438, -0.00438, 0.00657, 0.01971, 0.01752,
		0.00052, -0.00052, 0.00079, 0.00236, 0.00210,
		-0.01296, 0.01296, -0.01944, -0.05831, -0.05183,
		-0.01786, 0.01786, -0.02679, -0.08037, -0.07144,
	}, 1.0e-05) {
		t.Error("WForRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WFor.Grad().Data(), []float64{
		-0.00063, -0.00071, -0.00071, 0.00079,
		0.01752, 0.01971, 0.01971, -0.02189,
		0.00210, 0.00236, 0.00236, -0.00262,
		-0.05183, -0.05831, -0.05831, 0.06479,
		-0.07144, -0.08037, -0.08037, 0.08930,
	}, 1.0e-05) {
		t.Error("WFor doesn't match the expected values")
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
	model.BCand.Value().SetData([]float64{0.2, 0.0, -0.9, 0.7, -0.3})
	return model
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

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{-0.109472457893732, 0.998622353363888}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	x2 := g.NewVariable(mat.NewVecDense([]float64{3.3, -2.0, 0.1}), true)
	_ = proc.Forward(x2)
	s2 := proc.LastState()

	if !floats.EqualApprox(s2.Y.Value().Data(), []float64{0.331268038964841, 0.573215090496922}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	s.Y.PropagateGrad(mat.NewVecDense([]float64{-0.052743370318635, 0.421946962549082}))
	s2.Y.PropagateGrad(mat.NewVecDense([]float64{-0.043178925862421, 0.345431406899369}))

	g.BackwardAll()

	if !floats.EqualApprox(x.Grad().Data(), []float64{-0.00020471646083, 0.000980786890846, -0.000569825073637}, 1.0e-05) {
		t.Error("The input gradients x don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{0.106092179481869, 0.139102385565363, -0.107968562632196}, 1.0e-05) {
		t.Error("The input gradients x2 don't match the expected values")
	}

	if !floats.EqualApprox(model.WFor.Grad().Data(), []float64{
		0.002304509991833, -0.001396672722323, 6.98336361161645e-05,
		0.170963737408504, -0.103614386308184, 0.005180719315409,
	}, 1.0e-05) {
		t.Error("WFor doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BFor.Grad().Data(), []float64{
		0.000698336361162, 0.051807193154092,
	}, 1.0e-05) {
		t.Error("BFor doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WIn.Grad().Data(), []float64{
		-0.011143008433136, 0.034212414972157, -0.001262031874638,
		-0.000217468464402, 0.000935094429002, -3.36315794357185e-05,
	}, 1.0e-05) {
		t.Error("WIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BIn.Grad().Data(), []float64{
		-0.003648541266992, -7.38529540698525e-05,
	}, 1.0e-05) {
		t.Error("BIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		-0.081171280420682, 0.020093980778648, -0.001480107079882,
		0.695437596301958, -0.412559917587697, 0.020773676397031,
	}, 1.0e-05) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BCand.Grad().Data(), []float64{
		-0.024309231617813, 0.210650374323232,
	}, 1.0e-05) {
		t.Error("BCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WForRec.Grad().Data(), []float64{
		-7.64485978929299e-05, 0.000697374300423,
		-0.005671460771154, 0.051735821148717,
	}, 1.0e-05) {
		t.Error("WForRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WInRec.Grad().Data(), []float64{
		0.000890496046397, -0.008123223636552,
		2.24510905207517e-05, -0.000204801840415,
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
	model.BCand.Value().SetData([]float64{0.2, -0.1})
	return model
}
