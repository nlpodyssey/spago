// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestELUForward(t *testing.T) {
	t.Run("float32", testELUForward[float32])
	t.Run("float64", testELUForward[float64])
}

func testELUForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewDense[T](mat.WithBacking([]T{0.1, -0.2, 0.3, 0.0})),
		grad:         nil,
		requiresGrad: true,
	}
	alpha := &variable{
		value:        mat.Scalar[T](2.0),
		grad:         nil,
		requiresGrad: false,
	}
	f := NewELU(x, alpha)
	assert.Equal(t, []*variable{x, alpha}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.1, -0.36253849, 0.3, 0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{-1.0, 0.5, 0.8, 0.0})))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-1.0, 0.8187307, 0.8, 0.0}, x.grad.Data(), 1.0e-6)
}