// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package srn

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/losses"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestModel_Forward(t *testing.T) {
	t.Run("float32", testModelForward[float32])
	t.Run("float64", testModelForward[float64])
}

func testModelForward[T mat.DType](t *testing.T) {
	model := newTestModel[T]()

	// == Forward
	x := ag.NewVariable(mat.NewVecDense([]T{-0.8, -0.9, -0.9, 1.0}), true)

	y := model.Forward(x)[0]

	assert.InDeltaSlice(t, []T{-0.39693, -0.79688, 0.0, 0.70137, -0.18775}, y.Value().Data(), 1.0e-05)

	// == Backward

	gold := ag.NewVariable(mat.NewVecDense([]T{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(y, gold, false)
	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{-1.32512, -0.55398, 1.0709, 0.5709}, x.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.65167, 0.73313, 0.73313, -0.81459,
		0.45167, 0.50813, 0.50813, -0.56459,
		-0.12, -0.135, -0.135, 0.15,
		0.38151, 0.4292, 0.4292, -0.47689,
		0.49221, 0.55374, 0.55374, -0.61527,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.81459, -0.56459, 0.15, -0.47689, -0.61527,
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

	// == Forward
	x := ag.NewVariable(mat.NewVecDense([]T{-0.8, -0.9, -0.9, 1.0}), true)
	yPrev := ag.Tanh(ag.NewVariable(mat.NewVecDense([]T{-0.2, 0.2, -0.3, -0.9, -0.8}), true))

	s1 := model.Next(&State{Y: yPrev}, x)

	assert.InDeltaSlice(t, []T{0.59539, -0.8115, 0.17565, 0.88075, 0.08444}, s1.Y.Value().Data(), 1.0e-05)

	// == Backward

	gold := ag.NewVariable(mat.NewVecDense([]T{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(s1.Y, gold, false)
	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{-0.42553, -0.20751, 0.28232, 0.30119}, x.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		-0.01311, -0.01475, -0.01475, 0.01639,
		0.42655, 0.47987, 0.47987, -0.53319,
		-0.25248, -0.28404, -0.28404, 0.3156,
		0.13623, 0.15326, 0.15326, -0.17029,
		0.29036, 0.32666, 0.32666, -0.36295,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.01639, -0.53319, 0.3156, -0.17029, -0.36295,
	}, model.B.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.00323, 0.00323, -0.00477, -0.01174, -0.01088,
		0.10524, -0.10524, 0.15533, 0.38193, 0.35406,
		-0.06229, 0.06229, -0.09194, -0.22606, -0.20957,
		0.03361, -0.03361, 0.04961, 0.12198, 0.11308,
		0.07164, -0.07164, 0.10573, 0.25998, 0.24101,
	}, model.WRec.Grad().Data(), 1.0e-05)
}

func newTestModel[T mat.DType]() *Model {
	params := New[T](4, 5)
	mat.SetData[T](params.W.Value(), []T{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
		0.4, 1.0, -0.7, 0.8,
	})
	mat.SetData[T](params.WRec.Value(), []T{
		0.0, 0.8, 0.8, -1.0, -0.7,
		-0.7, -0.8, 0.2, -0.7, 0.7,
		-0.9, 0.9, 0.7, -0.5, 0.5,
		0.0, -0.1, 0.5, -0.2, -0.8,
		-0.6, 0.6, 0.8, -0.1, -0.3,
	})
	mat.SetData[T](params.B.Value(), []T{0.4, 0.0, -0.3, 0.8, -0.4})
	return params
}

func TestModel_ForwardSeq(t *testing.T) {
	t.Run("float32", testModelForwardSeq[float32])
	t.Run("float64", testModelForwardSeq[float64])
}

func testModelForwardSeq[T mat.DType](t *testing.T) {
	model := newTestModel2[T]()

	// == Forward

	s0 := &State{
		Y: ag.NewVariable(mat.NewVecDense([]T{0.0, 0.0}), true),
	}
	x := ag.NewVariable(mat.NewVecDense([]T{3.5, 4.0, -0.1}), true)
	s1 := model.Next(s0, x)

	assert.InDeltaSlice(t, []T{-0.9732261643, 0.9987757968}, s1.Y.Value().Data(), 1.0e-05)

	x2 := ag.NewVariable(mat.NewVecDense([]T{3.3, -2.0, 0.1}), true)
	s2 := model.Next(s1, x2)

	assert.InDeltaSlice(t, []T{-0.3773622668, 0.9671682519}, s2.Y.Value().Data(), 1.0e-05)

	// == Backward

	s1.Y.AccGrad(mat.NewVecDense([]T{-0.0548928728, 0.4391429825}))
	s2.Y.AccGrad(mat.NewVecDense([]T{-0.0527838886, 0.4222711092}))

	ag.Backward(s2.Y)

	assert.InDeltaSlice(t, []T{0.0015892366, 0.0013492153, -0.001893466}, x.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.0308715656, 0.019034727, -0.0223609451}, x2.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.1627089173, 0.0753041857, -0.0041459718,
		0.0936208886, -0.0504066848, 0.0026237982,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.0490749674, 0.0283072608,
	}, model.B.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.0440553621, -0.0452119261,
		-0.0265424287, 0.0272392341,
	}, model.WRec.Grad().Data(), 1.0e-05)
}

func newTestModel2[T mat.DType]() *Model {
	model := New[T](3, 2)
	mat.SetData[T](model.W.Value(), []T{
		-0.2, -0.3, 0.5,
		0.8, 0.2, 0.01,
	})
	mat.SetData[T](model.WRec.Value(), []T{
		0.5, 0.3,
		0.2, -0.1,
	})
	mat.SetData[T](model.B.Value(), []T{-0.2, 0.1})
	return model
}
