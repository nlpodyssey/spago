// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestLeakyReLUForward(t *testing.T) {
	t.Run("float32", testLeakyReLUForward[float32])
	t.Run("float64", testLeakyReLUForward[float64])
}

func testLeakyReLUForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value:        mat.NewVecDense([]T{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	alpha := &variable[T]{
		value:        mat.NewScalar[T](2.0),
		grad:         nil,
		requiresGrad: false,
	}
	f := NewLeakyReLU(x, alpha)
	assert.Equal(t, []*variable[T]{x, alpha}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{0.1, -0.4, 0.3, 0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []T{-1.0, 1.0, 0.8, 0.0}, x.grad.Data(), 1.0e-6)
}
