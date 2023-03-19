// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestSoftmax_Forward(t *testing.T) {
	t.Run("float32", testSoftmaxForward[float32])
	t.Run("float64", testSoftmaxForward[float64])
}

func testSoftmaxForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{-0.41, -1.08, 0, 0.87, -0.19, -0.75}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSoftmax(x)
	assert.Equal(t, []*variable{x}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{0.1166451, 0.0596882, 0.1757629, 0.4195304, 0.1453487, 0.083024}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{0.0, 0.0, -5.689482, 0.0, 0.0, 0.0}))

	assert.InDeltaSlice(t, []T{0.1166451, 0.0596882, -0.8242370, 0.4195304, 0.1453487, 0.083024}, x.grad.Data(), 1.0e-6)
}
