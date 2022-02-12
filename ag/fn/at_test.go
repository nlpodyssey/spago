// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestAt_Forward(t *testing.T) {
	t.Run("float32", testAtForward[float32])
	t.Run("float64", testAtForward[float64])
}

func testAtForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value: mat.NewDense(3, 4, []T{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewAt[T](x, 2, 3)
	y := f.Forward()

	assert.InDeltaSlice(t, []T{-0.1}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{0.5}))

	assert.InDeltaSlice(t, []T{
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.5,
	}, x.grad.Data(), 1.0e-6)
}
