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

func TestAtVec_Forward(t *testing.T) {
	t.Run("float32", testAtVecForward[float32])
	t.Run("float64", testAtVecForward[float64])
}

func testAtVecForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewAtVec(x, 1)
	assert.Equal(t, []*variable{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.2}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{0.5}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.0, 0.5, 0.0, 0.0}, x.grad.Data(), 1.0e-6)
}
