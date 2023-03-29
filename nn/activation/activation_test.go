// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/stretchr/testify/assert"
)

func TestModelReLU_Forward(t *testing.T) {
	t.Run("float32", testModelReLUForward[float32])
	t.Run("float64", testModelReLUForward[float64])
}

func testModelReLUForward[T float.DType](t *testing.T) {
	m := New(ReLU)

	x := mat.NewVecDense([]T{0.1, -0.2, 0.3, 0.0}, mat.WithGrad(true))
	y := m.Forward(x)[0]

	assert.InDeltaSlice(t, []T{0.1, 0.0, 0.3, 0.0}, y.Value().Data(), 1.0e-05)

	// == Backward
	y.AccGrad(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	ag.Backward(y)

	assert.InDeltaSlice(t, []T{-1.0, 0.0, 0.8, 0.0}, x.Grad().Data(), 1.0e-6)
}

func TestModelSwish_Forward(t *testing.T) {
	t.Run("float32", testModelSwishForward[float32])
	t.Run("float64", testModelSwishForward[float64])
}

func testModelSwishForward[T float.DType](t *testing.T) {
	beta := nn.NewParam(mat.NewScalar(T(2)))
	m := New(SwishB, beta)

	// == Forward
	x := mat.NewVecDense([]T{0.1, -0.2, 0.3, 0.0}, mat.WithGrad(true))
	y := m.Forward(x)[0]

	assert.InDeltaSlice(t, []T{0.0549833997, -0.080262468, 0.1936968919, 0.0}, y.Value().Data(), 1.0e-6)

	// == Backward
	y.AccGrad(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	ag.Backward(y)

	assert.InDeltaSlice(t, []T{-0.5993373119, 0.1526040208, 0.6263414804, 0.0}, x.Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.0188025145}, beta.Grad().Data(), 1.0e-6)
}
