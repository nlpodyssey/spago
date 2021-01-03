// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deltarnn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9, 1.0}), true)
	y := nn.ToNode(proc.Forward(x))

	assert.InDeltaSlice(t, []mat.Float{0.287518, 0.06939, -0.259175, 0.20769, -0.263768}, y.Value().Data(), 1.0e-05)

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]mat.Float{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, y, gold, false)
	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{0.023319, 0.253729, -0.122248, 0.190719}, x.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []mat.Float{
		0.007571, 0.008517, 0.008517, -0.009463,
		0.00075, 0.000843, 0.000843, -0.000937,
		-0.025515, -0.028704, -0.028704, 0.031894,
		0.073215, 0.082367, 0.082367, -0.091519,
		-0.159192, -0.179091, -0.179091, 0.19899,
	}, model.W.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []mat.Float{-0.091124, -0.095413, -0.057328, -0.258489, -0.318553}, model.B.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{-0.0368, -0.039102, 0.008963, -0.194915, 0.071569}, model.BPart.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{0.074722, 0.104, -0.017198, -0.018094, -0.066896}, model.Beta1.Grad().Data(), 1.0e-06)

	if model.WRec.HasGrad() {
		t.Error("WRec doesn't match the expected values")
	}

	if model.Beta2.HasGrad() {
		t.Error("Beta2 gradients don't match the expected values")
	}

	if model.Alpha.HasGrad() {
		t.Error("Alpha gradients don't match the expected values")
	}
}

func TestModel_ForwardWithPrev(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)
	proc.SetInitialState(
		&State{Y: g.NewVariable(mat.NewVecDense([]mat.Float{-0.197375, 0.197375, -0.291313, -0.716298, -0.664037}), true)},
	)

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9, 1.0}), true)
	y := nn.ToNode(proc.Forward(x))

	assert.InDeltaSlice(t, []mat.Float{0.202158, 0.228591, -0.240679, -0.350224, -0.476828}, y.Value().Data(), 1.0e-05)

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]mat.Float{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(g, y, gold, false)
	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{-0.177606, 0.379355, -0.085751, 0.080693}, x.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []mat.Float{
		-0.029084, -0.03272, -0.03272, 0.036355,
		-0.010076, -0.011335, -0.011335, 0.012595,
		-0.01401, -0.015761, -0.015761, 0.017512,
		0.252961, 0.284582, 0.284582, -0.316202,
		-0.072206, -0.081232, -0.081232, 0.090257,
	}, model.W.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []mat.Float{
		0.000242, -0.000242, 0.000357, 0.000878, 0.000813,
		0.001752, -0.001752, 0.002586, 0.00636, 0.005896,
		0.011671, -0.011671, 0.017226, 0.042355, 0.039265,
		-0.075195, 0.075195, -0.110982, -0.272891, -0.252981,
		0.008445, -0.008445, 0.012464, 0.030647, 0.028411,
	}, model.WRec.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []mat.Float{-0.122505, -0.06991, -0.054249, -0.493489, -0.353592}, model.B.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{-0.06814, -0.0145, -0.001299, -0.413104, -0.041468}, model.BPart.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{0.100454, 0.076202, -0.016275, -0.034544, -0.074254}, model.Beta1.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{-0.135487, 0.002895, -0.009628, -0.251233, -0.097111}, model.Beta2.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{0.1111, -0.003156, -0.002889, -0.017586, -0.020393}, model.Alpha.Grad().Data(), 1.0e-06)
}

func newTestModel() *Model {
	model := New(4, 5)
	model.W.Value().SetData([]mat.Float{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
		0.4, 1.0, -0.7, 0.8,
	})
	model.WRec.Value().SetData([]mat.Float{
		0.0, 0.8, 0.8, -1.0, -0.7,
		-0.7, -0.8, 0.2, -0.7, 0.7,
		-0.9, 0.9, 0.7, -0.5, 0.5,
		0.0, -0.1, 0.5, -0.2, -0.8,
		-0.6, 0.6, 0.8, -0.1, -0.3,
	})
	model.B.Value().SetData([]mat.Float{0.4, 0.0, -0.3, 0.8, -0.4})
	model.BPart.Value().SetData([]mat.Float{0.9, -0.5, 0.4, -0.8, 0.2})
	model.Alpha.Value().SetData([]mat.Float{-0.5, -0.3, 0.3, 0.4, 0.1})
	model.Beta1.Value().SetData([]mat.Float{-0.3, -0.4, -0.4, -0.4, -0.4})
	model.Beta2.Value().SetData([]mat.Float{-0.4, -0.2, 1.0, -0.8, 0.1})
	return model
}

func TestModel_ForwardSeq(t *testing.T) {
	model := newTestModel2()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)
	proc.SetInitialState(
		&State{Y: g.NewVariable(mat.NewVecDense([]mat.Float{0.0, 0.0}), true)},
	)

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]mat.Float{3.5, 4.0, -0.1}), true)
	_ = proc.Forward(x)
	s := proc.LastState()

	assert.InDeltaSlice(t, []mat.Float{0.176979535, 0.7339353781}, s.Y.Value().Data(), 1.0e-05)

	x2 := g.NewVariable(mat.NewVecDense([]mat.Float{3.3, -2.0, 0.1}), true)
	_ = proc.Forward(x2)
	s2 := proc.LastState()

	assert.InDeltaSlice(t, []mat.Float{0.0060780253, 0.6727636037}, s2.Y.Value().Data(), 1.0e-05)

	// == Backward

	s.Y.PropagateGrad(mat.NewVecDense([]mat.Float{-0.007, 0.002}))
	s2.Y.PropagateGrad(mat.NewVecDense([]mat.Float{-0.003, 0.005}))

	g.BackwardAll()

	assert.InDeltaSlice(t, []mat.Float{0.0002377894, 0.000303171, -0.0004869122}, x.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{-1.86775074391114e-005, -0.0004428228, 0.000880455}, x2.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.0023714943, -0.0074048063, 0.0002727509,
		0.0015561577, -0.0006155606, 3.61293436816857e-005,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-0.0020131993, 0.0008960376,
	}, model.B.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-0.0009511704, 3.27926892016188e-005,
	}, model.BPart.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-9.61771440502762e-005, -0.0003988473,
		0.0002176601, 0.0009026378,
	}, model.WRec.Grad().Data(), 1.0e-05)
}

func newTestModel2() *Model {
	model := New(3, 2)
	model.W.Value().SetData([]mat.Float{
		-0.2, -0.3, 0.5,
		0.8, 0.2, 0.01,
	})
	model.WRec.Value().SetData([]mat.Float{
		0.5, 0.3,
		0.2, -0.1,
	})
	model.B.Value().SetData([]mat.Float{-0.2, 0.1})
	model.BPart.Value().SetData([]mat.Float{0.5, 0.3})
	model.Alpha.Value().SetData([]mat.Float{0.5, 0.4})
	model.Beta1.Value().SetData([]mat.Float{-1.0, 0.5})
	model.Beta2.Value().SetData([]mat.Float{0.3, 0.6})

	return model
}
