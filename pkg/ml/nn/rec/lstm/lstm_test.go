// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lstm

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

	if !floats.EqualApprox(s.Cell.Value().Data(), []float64{-0.15, -0.114, -0.459, 0.691, -0.401}, 0.005) {
		t.Error("The cell doesn't match the expected values")
	}

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{-0.13, -0.05, -0.05, 0.31, -0.09}, 0.005) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, s.Y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.12, -0.14, 0.03, 0.02}, 0.005) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WIn.Grad().Data(), []float64{
		-0.04, -0.05, -0.05, 0.05,
		-0.02, -0.03, -0.03, 0.03,
		0.0, 0.0, 0.0, 0.0,
		0.07, 0.08, 0.08, -0.09,
		-0.02, -0.02, -0.02, 0.02,
	}, 0.005) {
		t.Error("WIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BIn.Grad().Data(), []float64{
		0.05, 0.03, 0.0, -0.09, 0.02,
	}, 0.005) {
		t.Error("BIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WOut.Grad().Data(), []float64{
		-0.01, -0.01, -0.01, 0.01,
		-0.02, -0.02, -0.02, 0.02,
		0.0, 0.0, 0.0, 0.0,
		0.16, 0.18, 0.18, -0.2,
		-0.03, -0.03, -0.03, 0.04,
	}, 0.005) {
		t.Error("WOut doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BOut.Grad().Data(), []float64{
		0.01, 0.02, 0.0, -0.2, 0.04,
	}, 0.005) {
		t.Error("BOut doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		0.16, 0.18, 0.18, -0.2,
		0.05, 0.06, 0.06, -0.07,
		0.0, 0.0, 0.0, 0.0,
		0.01, 0.01, 0.01, -0.01,
		0.01, 0.01, 0.01, -0.01,
	}, 0.005) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BCand.Grad().Data(), []float64{
		-0.20, -0.07, 0.0, -0.01, -0.01,
	}, 0.005) {
		t.Error("BCand doesn't match the expected values")
	}

	if model.WInRec.HasGrad() {
		t.Error("WInRec doesn't match the expected values")
	}

	if model.WOutRec.HasGrad() {
		t.Error("WOutRec doesn't match the expected values")
	}

	if model.WForRec.HasGrad() {
		t.Error("WForRec doesn't match the expected values")
	}

	if model.WCandRec.HasGrad() {
		t.Error("WCandRec doesn't match the expected values")
	}

	if model.WFor.HasGrad() {
		t.Error("WFor doesn't match the expected values")
	}

	if model.BFor.HasGrad() {
		t.Error("BFor doesn't match the expected values")
	}
}

func TestModel_ForwardWithPrev(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	proc.SetInitialState(&State{
		Cell: g.NewVariable(mat.NewVecDense([]float64{0.8, -0.6, 1.0, 0.1, 0.1}), true),
		Y:    g.NewVariable(mat.NewVecDense([]float64{-0.2, 0.2, -0.3, -0.9, -0.8}), true),
	})

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	_ = proc.Forward(x)
	s := proc.LastState()

	if !floats.EqualApprox(s.Cell.Value().Data(), []float64{0.5649, -0.2888, 0.3185, 0.9031, -0.4346}, 0.005) {
		t.Error("The cell doesn't match the expected values")
	}

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{0.47, -0.05, 0.01, 0.48, -0.16}, 0.005) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, s.Y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.106, -0.055, 0.002, 0.058}, 0.005) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WIn.Grad().Data(), []float64{
		-0.003, -0.003, -0.003, 0.003,
		0.007, 0.007, 0.007, -0.008,
		0.001, 0.002, 0.002, -0.002,
		0.044, 0.05, 0.05, -0.055,
		-0.036, -0.041, -0.041, 0.046,
	}, 0.005) {
		t.Error("WIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BIn.Grad().Data(), []float64{
		0.003, -0.008, -0.002, -0.055, 0.046,
	}, 0.005) {
		t.Error("BIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WOut.Grad().Data(), []float64{
		0.003, 0.004, 0.004, -0.004,
		-0.027, -0.03, -0.03, 0.033,
		-0.002, -0.002, -0.002, 0.002,
		0.146, 0.164, 0.164, -0.182,
		-0.047, -0.053, -0.053, 0.059,
	}, 0.005) {
		t.Error("WOut doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BOut.Grad().Data(), []float64{
		-0.004, 0.033, 0.002, -0.182, 0.059,
	}, 0.005) {
		t.Error("BOut doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		0.038, 0.043, 0.043, -0.048,
		0.024, 0.027, 0.027, -0.03,
		0.00, 0.00, 0.00, 0.00,
		0.005, 0.006, 0.006, -0.006,
		0.012, 0.013, 0.013, -0.015,
	}, 0.005) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BCand.Grad().Data(), []float64{
		-0.048, -0.03, 0.0, -0.006, -0.015,
	}, 0.005) {
		t.Error("BCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WInRec.Grad().Data(), []float64{
		-0.001, 0.001, -0.001, -0.003, -0.003,
		0.002, -0.002, 0.002, 0.007, 0.007,
		0.0, 0.0, 0.001, 0.002, 0.001,
		0.011, -0.011, 0.017, 0.05, 0.044,
		-0.009, 0.009, -0.014, -0.041, -0.036,
	}, 0.005) {
		t.Error("WInRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WOutRec.Grad().Data(), []float64{
		0.001, -0.001, 0.001, 0.004, 0.003,
		-0.007, 0.007, -0.01, -0.03, -0.027,
		0.0, 0.0, -0.001, -0.002, -0.002,
		0.036, -0.036, 0.055, 0.164, 0.146,
		-0.012, 0.012, -0.018, -0.053, -0.047,
	}, 0.005) {
		t.Error("WOutRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WForRec.Grad().Data(), []float64{
		0.001, -0.001, 0.001, 0.004, 0.004,
		-0.004, 0.004, -0.006, -0.017, -0.015,
		0.0, 0.0, 0.0, -0.001, -0.001,
		0.001, -0.001, 0.001, 0.003, 0.003,
		0.001, -0.001, 0.001, 0.004, 0.004,
	}, 0.005) {
		t.Error("WForRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCandRec.Grad().Data(), []float64{
		0.01, -0.01, 0.014, 0.043, 0.038,
		0.006, -0.006, 0.009, 0.027, 0.024,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.001, -0.001, 0.002, 0.006, 0.005,
		0.003, -0.003, 0.004, 0.013, 0.012,
	}, 0.005) {
		t.Error("WCandRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WFor.Grad().Data(), []float64{
		0.004, 0.004, 0.004, -0.005,
		-0.015, -0.017, -0.017, 0.019,
		-0.001, -0.001, -0.001, 0.001,
		0.003, 0.003, 0.003, -0.003,
		0.004, 0.004, 0.004, -0.005,
	}, 0.005) {
		t.Error("WFor doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BFor.Grad().Data(), []float64{
		-0.005, 0.019, 0.001, -0.003, -0.005,
	}, 0.005) {
		t.Error("BFor doesn't match the expected values")
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
	model.WOut.Value().SetData([]float64{
		0.1, 0.4, -1.0, 0.4,
		0.7, -0.2, 0.1, 0.0,
		0.7, 0.8, -0.5, -0.3,
		-0.9, 0.9, -0.3, -0.3,
		-0.7, 0.6, -0.6, -0.8,
	})
	model.WOutRec.Value().SetData([]float64{
		0.1, -0.6, -1.0, -0.1, -0.4,
		0.5, -0.9, 0.0, 0.8, 0.3,
		-0.3, -0.9, 0.3, 1.0, -0.2,
		0.7, 0.2, 0.3, -0.4, -0.6,
		-0.2, 0.5, -0.2, -0.9, 0.4,
	})
	model.BOut.Value().SetData([]float64{0.9, 0.2, -0.9, 0.2, -0.9})
	model.WFor.Value().SetData([]float64{
		-1.0, 0.2, 0.0, 0.2,
		-0.7, 0.7, -0.3, -0.3,
		0.3, -0.6, 0.0, 0.7,
		-1.0, -0.6, 0.9, 0.8,
		0.5, 0.8, -0.9, -0.8,
	})
	model.WForRec.Value().SetData([]float64{
		0.2, -0.3, -0.3, -0.5, -0.7,
		0.4, -0.1, -0.6, -0.4, -0.8,
		0.6, 0.6, 0.1, 0.7, -0.4,
		-0.8, 0.9, 0.1, -0.1, -0.2,
		-0.5, -0.3, -0.6, -0.6, 0.1,
	})
	model.BFor.Value().SetData([]float64{0.5, -0.5, 1.0, 0.4, 0.9})
	model.WCand.Value().SetData([]float64{
		0.2, 0.6, 0.0, 0.1,
		0.1, -0.3, -0.8, -0.5,
		-0.1, 0.0, 0.4, -0.4,
		-0.8, -0.3, -0.7, 0.3,
		-0.4, 0.9, 0.8, -0.3,
	})
	model.WCandRec.Value().SetData([]float64{
		-0.3, 0.3, -0.1, 0.6, -0.7,
		-0.2, -0.8, -0.6, -0.5, -0.4,
		-0.4, 0.8, -0.5, -0.1, 0.9,
		0.3, 0.7, 0.3, 0.0, -0.4,
		-0.3, 0.3, -0.7, 0.0, 0.7,
	})
	model.BCand.Value().SetData([]float64{0.2, -0.9, -0.9, 0.5, 0.1})
	return model
}

func TestModel_ForwardSeq(t *testing.T) {
	model := newTestModel2()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	proc.SetInitialState(&State{
		Cell: g.NewVariable(mat.NewVecDense([]float64{0.0, 0.0}), true),
		Y:    g.NewVariable(mat.NewVecDense([]float64{0.0, 0.0}), true),
	})

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{3.5, 4.0, -0.1}), true)
	_ = proc.Forward(x)
	s := proc.LastState()

	if !floats.EqualApprox(s.Cell.Value().Data(), []float64{-0.07229, 0.97534}, 1.0e-05) {
		t.Error("The cell doesn't match the expected values")
	}

	if !floats.EqualApprox(s.Y.Value().Data(), []float64{-0.00568, 0.64450}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	x2 := g.NewVariable(mat.NewVecDense([]float64{3.3, -2.0, 0.1}), true)
	_ = proc.Forward(x2)
	s2 := proc.LastState()

	if !floats.EqualApprox(s2.Cell.Value().Data(), []float64{0.39238, 0.99174}, 1.0e-05) {
		t.Error("The cell doesn't match the expected values")
	}

	if !floats.EqualApprox(s2.Y.Value().Data(), []float64{0.01688, 0.57555}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	s.Y.PropagateGrad(mat.NewVecDense([]float64{-0.045417243, 0.363337947}))
	s2.Y.PropagateGrad(mat.NewVecDense([]float64{-0.043997875, 0.351983003}))

	g.BackwardAll()

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.017677422, 0.001052328, -0.013964347}, 1.0e-05) {
		t.Error("The input gradients x don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{0.073384228, 0.058574837, -0.065843263}, 1.0e-05) {
		t.Error("The input gradients x2 don't match the expected values")
	}

	if !floats.EqualApprox(model.WIn.Grad().Data(), []float64{
		-0.0007430347, 0.0013851366, -5.39851412321307e-005,
		0.0261676128, 0.0125034523, -0.000161823,
	}, 1.0e-05) {
		t.Error("WIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BIn.Grad().Data(), []float64{
		-0.0002344176, 0.0076487617,
	}, 1.0e-05) {
		t.Error("BIn doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WOut.Grad().Data(), []float64{
		-0.0020964178, 0.0016978467, -7.79118487459307e-005,
		0.2587774428, 0.0141980025, 0.0020842003,
	}, 1.0e-05) {
		t.Error("WOut doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BOut.Grad().Data(), []float64{
		-0.0006395087, 0.0767240127,
	}, 1.0e-05) {
		t.Error("BOut doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		-0.0009563009, -0.0002035453, -2.61630702758865e-006,
		0.3072443936, -0.1849936394, 0.0092695324,
	}, 1.0e-05) {
		t.Error("WCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BCand.Grad().Data(), []float64{
		-0.0002820345, 0.0930923312,
	}, 1.0e-05) {
		t.Error("BCand doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WInRec.Grad().Data(), []float64{
		0.000002199, -0.000249511,
		-0.000017127, 0.0019433607,
	}, 1.0e-05) {
		t.Error("WInRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WOutRec.Grad().Data(), []float64{
		0.000004029, -0.0004571578,
		-0.000277093, 0.0314410032,
	}, 1.0e-05) {
		t.Error("WOutRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WForRec.Grad().Data(), []float64{
		-1.25423475053617e-007, 1.42314671841757e-005,
		-0.0001255625, 0.0142472443,
	}, 1.0e-05) {
		t.Error("WForRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WCandRec.Grad().Data(), []float64{
		8.7529846089037e-007, -9.931778175653e-005,
		-0.0005276474, 0.0598707467,
	}, 1.0e-05) {
		t.Error("WCandRec doesn't match the expected values")
	}

	if !floats.EqualApprox(model.WFor.Grad().Data(), []float64{
		7.28678234345139e-005, -4.41623172330387e-005, 2.20811586165194e-006,
		0.0729486052, -0.0442112759, 0.0022105638,
	}, 1.0e-05) {
		t.Error("WFor doesn't match the expected values")
	}

	if !floats.EqualApprox(model.BFor.Grad().Data(), []float64{
		2.20811586165194e-005, 0.0221056379,
	}, 1.0e-05) {
		t.Error("BFor doesn't match the expected values")
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
	model.WOut.Value().SetData([]float64{
		-0.7, 0.2, 0.1,
		0.5, 0.0, -0.5,
	})
	model.WOutRec.Value().SetData([]float64{
		0.2, 0.7,
		0.1, -0.7,
	})
	model.BOut.Value().SetData([]float64{-0.8, 0.0})
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
	model.WCandRec.Value().SetData([]float64{
		0.2, 0.7,
		0.1, -0.1,
	})
	model.BCand.Value().SetData([]float64{0.4, 0.3})
	return model
}
