// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDropout_Forward(t *testing.T) {
	t.Run("float32", testDropoutForward[float32])
	t.Run("float64", testDropoutForward[float64])
}

func testDropoutForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value:        mat.NewVecDense([]T{0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewDropout[T](x, 0.25, rand.NewLockedRand[T](1))
	assert.Equal(t, []*variable[T]{x}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{
		0.666666, 0.799999, -1.066666, -0.799999, 0.0, -0.5333333, 0.133333, 0.0, 0.399999, -0.666666,
	}, y.Data(), 1.0e-5)

	f.Backward(mat.NewVecDense([]T{0.5, 0.4, 0.2, -0.6, 0.3, 0.4, -0.8, -0.3, 0.0, 0.1}))

	assert.InDeltaSlice(t, []T{
		0.666666666, 0.533333333, 0.2666666, -0.8, 0.0, 0.5333333, -1.0666666, 0.0, 0.0, 0.133333,
	}, x.grad.Data(), 1.0e-6)
}

func TestZeroDropout_Forward(t *testing.T) {
	t.Run("float32", testZeroDropoutForward[float32])
	t.Run("float64", testZeroDropoutForward[float64])
}

func testZeroDropoutForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value:        mat.NewVecDense([]T{0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewDropout[T](x, 0.0, rand.NewLockedRand[T](1))
	assert.Equal(t, []*variable[T]{x}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{
		0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5,
	}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{0.5, 0.4, 0.2, -0.6, 0.3, 0.4, -0.8, -0.3, 0.0, 0.1}))

	assert.InDeltaSlice(t, []T{
		0.5, 0.4, 0.2, -0.6, 0.3, 0.4, -0.8, -0.3, 0.0, 0.1,
	}, x.grad.Data(), 1.0e-6)
}

func TestTotalDropout_Forward(t *testing.T) {
	t.Run("float32", testTotalDropoutForward[float32])
	t.Run("float64", testTotalDropoutForward[float64])
}

func testTotalDropoutForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value:        mat.NewVecDense([]T{0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewDropout[T](x, 1.0, rand.NewLockedRand[T](1))
	y := f.Forward()

	assert.InDeltaSlice(t, []T{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{0.5, 0.4, 0.2, -0.6, 0.3, 0.4, -0.8, -0.3, 0.0, 0.1}))

	assert.InDeltaSlice(t, []T{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	}, x.grad.Data(), 1.0e-6)
}
