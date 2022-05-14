// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestReduceSum_Forward(t *testing.T) {
	t.Run("float32", testReduceSumForward[float32])
	t.Run("float64", testReduceSumForward[float64])
}

func testReduceSumForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewReduceSum(x)
	assert.Equal(t, []*variable[T]{x}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{0.6}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{0.5}))

	assert.InDeltaSlice(t, []T{0.5, 0.5, 0.5, 0.5}, x.grad.Data(), 1.0e-6)
}
