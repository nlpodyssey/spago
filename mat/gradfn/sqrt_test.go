// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"math"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestSqrt_Forward(t *testing.T) {
	t.Run("float32", testSqrtForward[float32])
	t.Run("float64", testSqrtForward[float64])
}

func testSqrtForward[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{4, 9, 0}), mat.WithGrad(true))

	f := NewSqrt(x)
	assert.Equal(t, []mat.Tensor{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	mat.RequireMatrixInDelta(t, mat.NewDense[T](mat.WithBacking([]T{2, 3, 0})), y.(mat.Matrix), 1e-06)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{10, 20, 30})))
	assert.Nil(t, err)
	mat.RequireMatrixInDelta(t, mat.NewDense[T](mat.WithBacking([]T{2.5, 3.3333333, T(math.Inf(1))})), x.Grad().(mat.Matrix), 1e-06)
}
