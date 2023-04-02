// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

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
	x := newDualValue(mat.NewVecDense([]T{4, 9, 0}))

	f := NewSqrt(x)
	assert.Equal(t, []*variable{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	mat.RequireMatrixInDelta(t, mat.NewVecDense([]T{2, 3, 0}), y, 1e-06)

	err = f.Backward(mat.NewVecDense([]T{10, 20, 30}))
	assert.Nil(t, err)
	mat.RequireMatrixInDelta(t, mat.NewVecDense([]T{2.5, 3.3333333, T(math.Inf(1))}), x.grad, 1e-06)
}
