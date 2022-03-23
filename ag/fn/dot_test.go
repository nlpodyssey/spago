// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDot_Forward(t *testing.T) {
	t.Run("float32", testDotForward[float32])
	t.Run("float64", testDotForward[float64])
}

func testDotForward[T mat.DType](t *testing.T) {

	x1 := &variable[T]{
		value: mat.NewDense(3, 4, []T{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	x2 := &variable[T]{
		value: mat.NewDense(3, 4, []T{
			0.1, 0.8, 0.3, 0.1,
			0.1, -0.5, -0.9, 0.2,
			-0.2, 0.3, -0.4, -0.5,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewDot[T](x1, x2)
	assert.Equal(t, []*variable[T]{x1, x2}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{1.44}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{0.5}))

	assert.InDeltaSlice(t, []T{
		0.05, 0.4, 0.15, 0.05,
		0.05, -0.25, -0.45, 0.1,
		-0.1, 0.15, -0.2, -0.25,
	}, x1.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{
		0.05, 0.1, 0.15, 0.0,
		0.2, 0.25, -0.3, 0.35,
		-0.25, 0.4, -0.4, -0.05,
	}, x2.grad.Data(), 1.0e-6)
}
