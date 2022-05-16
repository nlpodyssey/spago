// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linear

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/losses"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/stretchr/testify/assert"
)

func TestModel_Forward(t *testing.T) {
	t.Run("float32", testModelForward[float32])
	t.Run("float64", testModelForward[float64])
}

type testLinearWithActivationModel struct {
	nn.Module
	M1 *Model
	M2 *activation.Model
}

func (m *testLinearWithActivationModel) forward(x ag.Node) ag.Node {
	return m.M2.Forward(m.M1.Forward(x)[0])[0]
}

func testModelForward[T float.DType](t *testing.T) {
	model := newTestModel[T]()

	m := &testLinearWithActivationModel{
		M1: model,
		M2: activation.New(activation.Tanh),
	}

	// == Forward

	x := ag.Var(mat.NewVecDense([]T{-0.8, -0.9, -0.9, 1.0})).WithGrad(true)
	y := m.forward(x)

	assert.InDeltaSlice(t, []T{-0.39693, -0.79688, 0.0, 0.70137, -0.18775}, y.Value().Data(), 1.0e-05)

	// == Backward

	gold := ag.Var(mat.NewVecDense([]T{0.0, 0.5, -0.4, -0.9, 0.9}))
	loss := losses.MSE(y, gold, false)
	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.0126, -2.07296, 1.07476, -0.14158}, x.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		0.26751, 0.30095, 0.30095, -0.33439,
		0.37867, 0.42601, 0.42601, -0.47334,
		-0.32, -0.36, -0.36, 0.4,
		-0.65089, -0.73226, -0.73226, 0.81362,
		0.83952, 0.94446, 0.94446, -1.04940,
	}, model.W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.33439, -0.47334, 0.4, 0.81362, -1.0494,
	}, model.B.Grad().Data(), 1.0e-05)
}

func newTestModel[T float.DType]() *Model {
	model := New[T](4, 5)
	mat.SetData[T](model.W.Value(), []T{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
		0.4, 1.0, -0.7, 0.8,
	})
	mat.SetData[T](model.B.Value(), []T{0.4, 0.0, -0.3, 0.8, -0.4})
	return model
}
