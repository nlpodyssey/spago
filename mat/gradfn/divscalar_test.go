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

func TestScalarDiv_Forward(t *testing.T) {
	t.Run("float32", testScalarDivForward[float32])
	t.Run("float64", testScalarDivForward[float64])
}

func testScalarDivForward[T float.DType](t *testing.T) {
	x1 := mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 0.3, 0.0}), mat.WithGrad(true))
	x2 := mat.Scalar[T](2.0)

	f := NewDivScalar(x1, x2)
	assert.Equal(t, []mat.Tensor{x1, x2}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.05, 0.1, 0.15, 0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{-1.0, 0.5, 0.8, 0.0})))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-0.5, 0.25, 0.4, 0.0}, x1.Grad().Data(), 1.0e-6)
}
