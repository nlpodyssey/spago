// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ltm

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

	if !floats.EqualApprox(s.Cell.Value().Data(), []float64{0.531299, 0.439948, 0.484336, 0.443710}, 1.0e-06) {
		t.Error("The cell doesn't match the expected values")
	}

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{0.368847, 0.208984, 0.354078, 0.350904}, 1.0e-06) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64}), false)
	loss := losses.MSE(g, s.Y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.22696676, 0.009912126, -0.105133662, -0.040795301}, 1.0e-06) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.W1.Grad().Data(), []float64{
		-0.021337426, -0.024004605, -0.024004605, 0.026671783,
		-0.004265735, -0.004798952, -0.004798952, 0.005332169,
		-0.012587706, -0.01416117, -0.01416117, 0.015734633,
		0.007898459, 0.008885766, 0.008885766, -0.009873074,
	}, 1.0e-06) {
		t.Error("W1 doesn't match the expected values")
	}

	if !floats.EqualApprox(model.W2.Grad().Data(), []float64{
		-0.009138756, -0.0102811, -0.0102811, 0.011423445,
		-0.003507654, -0.00394611, -0.00394611, 0.004384567,
		-0.02235721, -0.025151862, -0.025151862, 0.027946513,
		0.008675311, 0.009759725, 0.009759725, -0.010844138,
	}, 1.0e-06) {
		t.Error("W2 doesn't match the expected values")
	}

	if !floats.EqualApprox(model.W3.Grad().Data(), []float64{
		0.018148829, 0.020417433, 0.020417433, -0.022686037,
		0.047484924, 0.053420539, 0.053420539, -0.059356155,
		-0.0384011, -0.043201237, -0.043201237, 0.048001375,
		0.075690387, 0.085151686, 0.085151686, -0.094612894,
	}, 1.0e-06) {
		t.Error("W3 doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCell.Grad().Data(), []float64{
		-0.00747137, -0.003365414, -0.004877438, -0.008458711,
		-0.013604532, -0.006128044, -0.008881272, -0.015402369,
		0.019774006, 0.008907031, 0.01290881, 0.022387139,
		-0.054063761, -0.024352556, -0.035293749, -0.061208281,
	}, 1.0e-06) {
		t.Error("WCell doesn't match the expected values")
	}
}

func TestModel_ForwardWithPrev(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	yPrev := g.NewVariable(mat.NewVecDense([]float64{0.7, 0.6, 0.2, 0.8}), true)
	cellPrev := g.NewVariable(mat.NewVecDense([]float64{0.574443, 0.425557, 0.401312, 0.524979}), true)
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	proc.SetInitialState(&State{
		Cell: cellPrev,
		Y:    yPrev,
	})

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	_ = proc.Forward(x)
	s := proc.LastState()

	if !floats.EqualApprox(s.Cell.Value().Data(), []float64{0.633246, 0.276811, 0.474442, 0.256067}, 1.0e-06) {
		t.Error("The cell doesn't match the expected values")
	}

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{0.379117, 0.106466, 0.381340, 0.191636}, 1.0e-06) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64}), false)
	loss := losses.MSE(g, s.Y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.199768, 0.0135238, -0.0872397, -0.034826}, 1.0e-06) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(yPrev.Grad().Data(), []float64{0.199768, 0.0135238, -0.0872397, -0.034826}, 1.0e-06) {
		t.Error("The yPrev gradients don't match the expected values")
	}

	if !floats.EqualApprox(cellPrev.Grad().Data(), []float64{0.144275, 0.060885, 0.226774, -0.08242}, 1.0e-06) {
		t.Error("The cPrev gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.W1.Grad().Data(), []float64{
		-0.002465, -0.007395, -0.017255, 0.044372,
		-0.000447, -0.001341, -0.00312899, 0.008045,
		-0.0018103, -0.005430947, -0.0126722, 0.032585,
		0.000773, 0.002319, 0.0054115, -0.0139153,
	}, 1.0e-06) {
		t.Error("W1 doesn't match the expected values")
	}

	if !floats.EqualApprox(model.W2.Grad().Data(), []float64{
		-0.0007834, -0.0023503, -0.00548415, 0.014102,
		-0.0002864, -0.0008592, -0.0020049, 0.0051554,
		-0.00371106, -0.011133208, -0.02597748, 0.0667992,
		0.00097089, 0.00291269, 0.00679628, -0.017476,
	}, 1.0e-06) {
		t.Error("W2 doesn't match the expected values")
	}

	if !floats.EqualApprox(model.W3.Grad().Data(), []float64{
		0.0029041, 0.0087125, 0.0203292, -0.0522752,
		0.0042162, 0.01264882, 0.0295139, -0.075892,
		-0.0039761, -0.011928, -0.027832, 0.07157,
		0.0069838, 0.020951, 0.048887084, -0.1257096,
	}, 1.0e-06) {
		t.Error("W3 doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCell.Grad().Data(), []float64{
		-0.021923, -0.013695, -0.017628, -0.018937,
		-0.040928, -0.025568, -0.032911, -0.035353,
		0.087961, 0.054951, 0.0707317, 0.0759816,
		-0.170559, -0.106551, -0.13715, -0.147329,
	}, 1.0e-06) {
		t.Error("WCell doesn't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(4)
	model.W1.Value().SetData([]float64{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
	})
	model.W2.Value().SetData([]float64{
		0.1, 0.4, -1.0, 0.4,
		0.7, -0.2, 0.1, 0.0,
		0.7, 0.8, -0.5, -0.3,
		-0.9, 0.9, -0.3, -0.3,
	})
	model.W3.Value().SetData([]float64{
		-1.0, 0.2, 0.0, 0.2,
		-0.7, 0.7, -0.3, -0.3,
		0.3, -0.6, 0.0, 0.7,
		-1.0, -0.6, 0.9, 0.8,
	})
	model.WCell.Value().SetData([]float64{
		0.2, 0.6, 0.0, 0.1,
		0.1, -0.3, -0.8, -0.5,
		-0.1, 0.0, 0.4, -0.4,
		-0.8, -0.3, -0.7, 0.3,
	})
	return model
}

func TestModel_ForwardSeq(t *testing.T) { //TODO FIX TEST
	model := newTestModel2()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	proc.SetInitialState(&State{
		Cell: g.NewVariable(mat.NewVecDense([]float64{0.0, 0.0, 0.0}), true),
		Y:    g.NewVariable(mat.NewVecDense([]float64{0.0, 0.0, 0.0}), true),
	})

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{3.5, 4.0, -0.1}), true)
	_ = proc.Forward(x)
	s := proc.LastState()

	if !floats.EqualApprox(s.Cell.Value().Data(), []float64{0.6585208524, 0.5193369948, 0.3051361057}, 1.0e-05) {
		t.Error("The cell doesn't match the expected values")
	}

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{0.1052066064, 0.445668489, 0.0920091497}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	x2 := g.NewVariable(mat.NewVecDense([]float64{3.3, -2.0, 0.1}), true)
	_ = proc.Forward(x2)
	s2 := proc.LastState()

	if !floats.EqualApprox(s2.Cell.Value().Data(), []float64{0.7639620348, 0.5509132249, 0.1590346479}, 1.0e-05) {
		t.Error("The cell doesn't match the expected values")
	}

	if !floats.EqualApprox(s2.Y.Value().Data(), []float64{0.0492343522, 0.4588769062, 0.0947403852}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	s.Y.PropagateGrad(mat.NewVecDense([]float64{-0.2, -0.3, -0.4}))
	s2.Y.PropagateGrad(mat.NewVecDense([]float64{0.6, -0.3, -0.2}))

	g.BackwardAll()

	if !floats.EqualApprox(x.Grad().Data(), []float64{
		0.0058833, 0.0008477, -0.0048020}, 1.0e-05) {
		t.Error("The input gradients x don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{
		-0.02891086, 0.00700993, 0.0099656}, 1.0e-05) {
		t.Error("The input gradients x2 don't match the expected values")
	}

	if !floats.EqualApprox(model.W1.Grad().Data(), []float64{
		0.000588, 0.00006698, 0.00001534,
		0.006493, 0.000478661, 0.0001832,
		-0.007074, -0.0004098, -0.0002055,
	}, 1.0e-05) {
		t.Error("W1 doesn't match the expected values")
	}

	if !floats.EqualApprox(model.W2.Grad().Data(), []float64{
		0.000279, -0.00007709, 0.00001306,
		0.0241938, 0.006375, 0.000438,
		-0.013505, -0.0093022, 0.00006,
	}, 1.0e-05) {
		t.Error("W2 doesn't match the expected values")
	}

	if !floats.EqualApprox(model.W3.Grad().Data(), []float64{
		0.02328, -0.12389, 0.00733,
		-0.143141, -0.038342, -0.002563,
		-0.113814, -0.088355, 0.00103,
	}, 1.0e-05) {
		t.Error("W3 doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCell.Grad().Data(), []float64{
		0.005861, 0.001615, -0.001051,
		-0.06589, -0.122702, -0.080731,
		-0.01829, -0.041815, -0.02927,
	}, 1.0e-05) {
		t.Error("WCell doesn't match the expected values")
	}
}

func newTestModel2() *Model {
	model := New(3)
	model.W1.Value().SetData([]float64{
		-0.2, -0.3, 0.5,
		0.8, 0.2, 0.01,
		0.3, 0.4, -1.0,
	})
	model.W2.Value().SetData([]float64{
		0.3, 0.2, -0.4,
		0.4, 0.1, -0.6,
		0.2, 0.1, 0.3,
	})
	model.W3.Value().SetData([]float64{
		-0.7, 0.2, 0.1,
		0.5, 0.0, -0.5,
		0.0, -0.2, 0.4,
	})
	model.WCell.Value().SetData([]float64{
		0.5, 0.3, 0.5,
		0.2, -0.1, 0.2,
		-0.6, -0.9, 0.0,
	})
	return model
}
