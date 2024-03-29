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

func TestSoftmax_Forward(t *testing.T) {
	t.Run("float32", testSoftmaxForward[float32])
	t.Run("float64", testSoftmaxForward[float64])
}

func testSoftmaxForward[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{-0.41, -1.08, 0, 0.87, -0.19, -0.75}), mat.WithGrad(true))
	f := NewSoftmax(x)
	assert.Equal(t, []mat.Tensor{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.1166451, 0.0596882, 0.1757629, 0.4195304, 0.1453487, 0.083024}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{0.0, 0.0, -5.689482, 0.0, 0.0, 0.0})))
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.1166451, 0.0596882, -0.8242370, 0.4195304, 0.1453487, 0.083024}, x.Grad().Data(), 1.0e-6)
}
