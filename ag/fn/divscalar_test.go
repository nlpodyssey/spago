// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestScalarDiv_Forward(t *testing.T) {
	t.Run("float32", testScalarDivForward[float32])
	t.Run("float64", testScalarDivForward[float64])
}

func testScalarDivForward[T mat.DType](t *testing.T) {
	x1 := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	x2 := &variable{
		value:        mat.NewScalar[T](2.0),
		grad:         nil,
		requiresGrad: false,
	}
	f := NewDivScalar(x1, x2)
	assert.Equal(t, []*variable{x1, x2}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{0.05, 0.1, 0.15, 0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []T{-0.5, 0.25, 0.4, 0.0}, x1.grad.Data(), 1.0e-6)
}
