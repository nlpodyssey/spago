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

func TestRotateR_Forward(t *testing.T) {
	t.Run("float32", testRotateRForward[float32])
	t.Run("float64", testRotateRForward[float64])
}

func testRotateRForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewRotateR(x, 1)
	assert.Equal(t, []*variable{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}))
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{
		0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1,
	}, x.grad.Data(), 1.0e-6)
}
