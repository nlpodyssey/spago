// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"github.com/stretchr/testify/assert"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

func TestAt_Forward(t *testing.T) {
	t.Run("float32", testAtForward[float32])
	t.Run("float64", testAtForward[float64])
}

func testAtForward[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithShape(3, 4), mat.WithBacking([]T{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	}), mat.WithGrad(true))

	f := NewAt(x, 2, 3)
	assert.Equal(t, []mat.Tensor{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-0.1}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{0.5})))
	assert.NoError(t, err)

	assert.InDeltaSlice(t, []T{
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.5,
	}, x.Grad().Data(), 1.0e-6)
}
