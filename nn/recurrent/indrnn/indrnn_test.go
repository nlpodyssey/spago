// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package indrnn

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	t.Run("float32", testModelForward[float32])
	t.Run("float64", testModelForward[float64])
}

func testModelForward[T mat.DType](t *testing.T) {
	model := newTestModel[T]()
	g := ag.NewGraph[T](ag.WithMode[T](ag.Training))

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]T{-0.8, -0.9, -0.9, 1.0}), true)
	y := ag.Bind(g, model).Forward(x)[0]

	assert.InDeltaSlice(t, []T{-0.39693, -0.796878, 0.0, 0.701374, -0.187746}, y.Value().Data(), 1.0e-05)

	// == Backward

	g.Backward(y, ag.OutputGrad[T](mat.NewVecDense([]T{0.57, 0.75, -0.15, 1.64, 0.45})))

	assert.InDeltaSlice(t, []T{1.166963, -0.032159, -0.705678, -0.318121}, x.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.384155, -0.432175, -0.432175, 0.480194,
		-0.218991, -0.246365, -0.246365, 0.273739,
		0.120000, 0.135000, 0.135000, -0.150000,
		-0.666594, -0.749918, -0.749918, 0.833242,
		-0.347310, -0.390724, -0.390724, 0.434138,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.480194, 0.273739, -0.150000, 0.833242, 0.434138,
	}, model.B.Grad().Data(), 1.0e-05)

	if model.WRec.HasGrad() {
		t.Error("WRec doesn't match the expected values")
	}
}

func TestModel_ForwardWithPrev(t *testing.T) {
	t.Run("float32", testModelForwardWithPrev[float32])
	t.Run("float64", testModelForwardWithPrev[float64])
}

func testModelForwardWithPrev[T mat.DType](t *testing.T) {
	model := newTestModel[T]()
	g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
	proc := ag.Bind(g, model)

	// == Forward
	x := g.NewVariable(mat.NewVecDense([]T{-0.8, -0.9, -0.9, 1.0}), true)
	yPrev := ag.Tanh(g.NewVariable(mat.NewVecDense([]T{-0.2, 0.2, -0.3, -0.9, -0.8}), true))
	s1 := proc.Next(&State[T]{Y: yPrev}, x)

	assert.InDeltaSlice(t, []T{-0.39693, -0.842046, 0.256335, 0.701374, 0.205456}, s1.Y.Value().Data(), 1.0e-05)

	// == Backward

	g.Backward(s1.Y, ag.OutputGrad[T](mat.NewVecDense([]T{0.57, 0.75, -0.15, 1.64, 0.45})))

	assert.InDeltaSlice(t, []T{1.133745, -0.019984, -0.706080, -0.271285}, x.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		-0.384155, -0.432175, -0.432175, 0.480194,
		-0.174576, -0.196397, -0.196397, 0.218219,
		0.112115, 0.126129, 0.126129, -0.140144,
		-0.666594, -0.749918, -0.749918, 0.833242,
		-0.344804, -0.387904, -0.387904, 0.431005,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.480194, 0.218219, -0.140144, 0.833242, 0.431005,
	}, model.B.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.094779, 0.043071, 0.040826, -0.596849, -0.286203,
	}, model.WRec.Grad().Data(), 1.0e-05)
}

func newTestModel[T mat.DType]() *Model[T] {
	params := New[T](4, 5, activation.Tanh)
	params.W.Value().SetData([]T{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
		0.4, 1.0, -0.7, 0.8,
	})
	params.WRec.Value().SetData([]T{0.0, -0.7, -0.9, 0.0, -0.6})
	params.B.Value().SetData([]T{0.4, 0.0, -0.3, 0.8, -0.4})
	return params
}

func TestModel_ForwardSeq(t *testing.T) {
	t.Run("float32", testModelForwardSeq[float32])
	t.Run("float64", testModelForwardSeq[float64])
}

func testModelForwardSeq[T mat.DType](t *testing.T) {
	model := newTestModel2[T]()
	g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
	proc := ag.Bind(g, model)

	// == Forward

	s0 := &State[T]{Y: g.NewVariable(mat.NewVecDense([]T{0.0, 0.0}), true)}
	x := g.NewVariable(mat.NewVecDense([]T{3.5, 4.0, -0.1}), true)
	s1 := proc.Next(s0, x)

	assert.InDeltaSlice(t, []T{-0.9732261643, 0.9987757968}, s1.Y.Value().Data(), 1.0e-05)

	x2 := g.NewVariable(mat.NewVecDense([]T{3.3, -2.0, 0.1}), true)
	s2 := proc.Next(s1, x2)

	assert.InDeltaSlice(t, []T{-0.602213565, 0.9898794918}, s2.Y.Value().Data(), 1.0e-05)

	// == Backward

	s1.Y.PropagateGrad(mat.NewVecDense([]T{-0.007, 0.002}))
	s2.Y.PropagateGrad(mat.NewVecDense([]T{-0.003, 0.005}))

	g.BackwardAll()

	assert.InDeltaSlice(t, []T{8.79795806788067e-005, 0.0001270755, -0.0002101123}, x.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{0.0004629577, 0.0005937435, -0.0009550013}, x2.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.0077807832, 0.0021427428, -0.0001491694,
		0.0003494152, -0.0001818106, 9.5799118461089e-006,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.002332339, 0.0001055868,
	}, model.B.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.0018608245, 0.0001005697,
	}, model.WRec.Grad().Data(), 1.0e-05)
}

func newTestModel2[T mat.DType]() *Model[T] {
	model := New[T](3, 2, activation.Tanh)
	model.W.Value().SetData([]T{
		-0.2, -0.3, 0.5,
		0.8, 0.2, 0.01,
	})
	model.WRec.Value().SetData([]T{0.5, 0.3})
	model.B.Value().SetData([]T{-0.2, 0.1})
	return model
}
