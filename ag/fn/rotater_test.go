// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestRotateR_Forward(t *testing.T) {
	t.Run("float32", testRotateRForward[float32])
	t.Run("float64", testRotateRForward[float64])
}

func testRotateRForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewRotateR[T](x, 1)
	assert.Equal(t, []*variable[T]{x}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}))

	assert.InDeltaSlice(t, []T{
		0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1,
	}, x.grad.Data(), 1.0e-6)
}
