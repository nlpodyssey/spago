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

func TestRotateR_Forward(t *testing.T) {
	t.Run("float32", testRotateRForward[float32])
	t.Run("float64", testRotateRForward[float64])
}

func testRotateRForward[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}), mat.WithGrad(true))
	f := NewRotateR(x, 1)
	assert.Equal(t, []mat.Tensor{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8})))
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{
		0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1,
	}, x.Grad().Data(), 1.0e-6)
}
