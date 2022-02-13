// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sqrdist

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
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
	x := g.NewVariable(mat.NewVecDense([]T{0.3, 0.5, -0.4}), true)
	y := nn.ToNode[T](nn.Reify(model, g).Forward(x))

	assert.InDeltaSlice(t, []T{0.5928}, y.Value().Data(), 1.0e-05)

	// == Backward
	g.Backward(y, ag.OutputGrad[T](mat.NewScalar[T](-0.8)))

	assert.InDeltaSlice(t, []T{-0.9568, -0.848, 0.5936}, x.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.2976, -0.496, 0.3968,
		0.0144, 0.024, -0.0192,
		-0.1488, -0.248, 0.1984,
		-0.1584, -0.264, 0.2112,
		0.024, 0.04, -0.032,
	}, model.B.Grad().Data(), 1.0e-06)
}

func newTestModel[T mat.DType]() *Model[T] {
	model := New[T](3, 5)
	model.B.Value().SetData([]T{
		0.4, 0.6, -0.5,
		-0.5, 0.4, 0.2,
		0.5, 0.4, 0.1,
		0.5, 0.2, -0.2,
		-0.3, 0.4, 0.4,
	})
	return model
}
