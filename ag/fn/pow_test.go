// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestPow_Forward(t *testing.T) {
	t.Run("float32", testPowForward[float32])
	t.Run("float64", testPowForward[float64])
}

func testPowForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewPow[T](x, 3.0)
	assert.Equal(t, []*variable[T]{x}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{0.001, 0.008, 0.027, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []T{-0.03, 0.06, 0.216, 0}, x.grad.Data(), 1.0e-6)
}
