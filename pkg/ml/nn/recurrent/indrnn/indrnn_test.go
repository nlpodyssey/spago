// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package indrnn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9, 1.0}), true)
	y := nn.ToNode(nn.Reify(ctx, model).(*Model).Forward(x))

	assert.InDeltaSlice(t, []mat.Float{-0.39693, -0.796878, 0.0, 0.701374, -0.187746}, y.Value().Data(), 1.0e-05)

	// == Backward

	g.Backward(y, ag.OutputGrad(mat.NewVecDense([]mat.Float{0.57, 0.75, -0.15, 1.64, 0.45})))

	assert.InDeltaSlice(t, []mat.Float{1.166963, -0.032159, -0.705678, -0.318121}, x.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-0.384155, -0.432175, -0.432175, 0.480194,
		-0.218991, -0.246365, -0.246365, 0.273739,
		0.120000, 0.135000, 0.135000, -0.150000,
		-0.666594, -0.749918, -0.749918, 0.833242,
		-0.347310, -0.390724, -0.390724, 0.434138,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.480194, 0.273739, -0.150000, 0.833242, 0.434138,
	}, model.B.Grad().Data(), 1.0e-05)

	if model.WRec.HasGrad() {
		t.Error("WRec doesn't match the expected values")
	}
}

func TestModel_ForwardWithPrev(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9, 1.0}), true)
	yPrev := g.Tanh(g.NewVariable(mat.NewVecDense([]mat.Float{-0.2, 0.2, -0.3, -0.9, -0.8}), true))
	proc := nn.Reify(ctx, model).(*Model)
	proc.SetInitialState(&State{Y: yPrev})
	y := nn.ToNode(proc.Forward(x))

	assert.InDeltaSlice(t, []mat.Float{-0.39693, -0.842046, 0.256335, 0.701374, 0.205456}, y.Value().Data(), 1.0e-05)

	// == Backward

	g.Backward(y, ag.OutputGrad(mat.NewVecDense([]mat.Float{0.57, 0.75, -0.15, 1.64, 0.45})))

	assert.InDeltaSlice(t, []mat.Float{1.133745, -0.019984, -0.706080, -0.271285}, x.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []mat.Float{
		-0.384155, -0.432175, -0.432175, 0.480194,
		-0.174576, -0.196397, -0.196397, 0.218219,
		0.112115, 0.126129, 0.126129, -0.140144,
		-0.666594, -0.749918, -0.749918, 0.833242,
		-0.344804, -0.387904, -0.387904, 0.431005,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.480194, 0.218219, -0.140144, 0.833242, 0.431005,
	}, model.B.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-0.094779, 0.043071, 0.040826, -0.596849, -0.286203,
	}, model.WRec.Grad().Data(), 1.0e-05)
}

func newTestModel() *Model {
	params := New(4, 5, ag.OpTanh)
	params.W.Value().SetData([]mat.Float{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
		0.4, 1.0, -0.7, 0.8,
	})
	params.WRec.Value().SetData([]mat.Float{0.0, -0.7, -0.9, 0.0, -0.6})
	params.B.Value().SetData([]mat.Float{0.4, 0.0, -0.3, 0.8, -0.4})
	return params
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

	assert.InDeltaSlice(t, []mat.Float{-0.9732261643, 0.9987757968}, s.Y.Value().Data(), 1.0e-05)

	x2 := g.NewVariable(mat.NewVecDense([]mat.Float{3.3, -2.0, 0.1}), true)
	_ = proc.Forward(x2)
	s2 := proc.LastState()

	assert.InDeltaSlice(t, []mat.Float{-0.602213565, 0.9898794918}, s2.Y.Value().Data(), 1.0e-05)

	// == Backward

	s.Y.PropagateGrad(mat.NewVecDense([]mat.Float{-0.007, 0.002}))
	s2.Y.PropagateGrad(mat.NewVecDense([]mat.Float{-0.003, 0.005}))

	g.BackwardAll()

	assert.InDeltaSlice(t, []mat.Float{8.79795806788067e-005, 0.0001270755, -0.0002101123}, x.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{0.0004629577, 0.0005937435, -0.0009550013}, x2.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-0.0077807832, 0.0021427428, -0.0001491694,
		0.0003494152, -0.0001818106, 9.5799118461089e-006,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-0.002332339, 0.0001055868,
	}, model.B.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.0018608245, 0.0001005697,
	}, model.WRec.Grad().Data(), 1.0e-05)
}

func newTestModel2() *Model {
	model := New(3, 2, ag.OpTanh)
	model.W.Value().SetData([]mat.Float{
		-0.2, -0.3, 0.5,
		0.8, 0.2, 0.01,
	})
	model.WRec.Value().SetData([]mat.Float{0.5, 0.3})
	model.B.Value().SetData([]mat.Float{-0.2, 0.1})
	return model
}
