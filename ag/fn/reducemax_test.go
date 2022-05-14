// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestReduceMax_Forward(t *testing.T) {
	t.Run("float32", testReduceMaxForward[float32])
	t.Run("float64", testReduceMaxForward[float64])
}

func testReduceMaxForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewReduceMax(x)
	assert.Equal(t, []*variable[T]{x}, f.Operands())

	y := f.Forward()
	assert.InDeltaSlice(t, []T{0.3}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{0.5}))
	assert.InDeltaSlice(t, []T{0.0, 0.0, 0.5, 0.0}, x.grad.Data(), 1.0e-6)
}
