// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSwishBForward(t *testing.T) {
	t.Run("float32", testSwishBForward[float32])
	t.Run("float64", testSwishBForward[float64])
}

func testSwishBForward[T mat.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	beta := &variable{
		value:        mat.NewScalar[T](2.0),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSwishB(x, beta)
	assert.Equal(t, []*variable{x, beta}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{0.0549833997, -0.080262468, 0.1936968919, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []T{-0.5993373119, 0.1526040208, 0.6263414804, 0.0}, x.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.0188025145}, beta.grad.Data(), 1.0e-6)
}
