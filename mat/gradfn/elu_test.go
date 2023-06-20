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

func TestELUForward(t *testing.T) {
	t.Run("float32", testELUForward[float32])
	t.Run("float64", testELUForward[float64])
}

func testELUForward[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{0.1, -0.2, 0.3, 0.0}), mat.WithGrad(true))
	alpha := mat.Scalar[T](2.0)

	f := NewELU(x, alpha)
	assert.Equal(t, []mat.Tensor{x, alpha}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.1, -0.36253849, 0.3, 0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{-1.0, 0.5, 0.8, 0.0})))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-1.0, 0.8187307, 0.8, 0.0}, x.Grad().Data(), 1.0e-6)
}
