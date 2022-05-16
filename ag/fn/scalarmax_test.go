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

func TestScalarMax_Forward(t *testing.T) {
	t.Run("float32", testScalarMaxForward[float32])
	t.Run("float64", testScalarMaxForward[float64])
}

func testScalarMaxForward[T float.DType](t *testing.T) {
	xs := []*variable{
		{mat.NewScalar[T](2.0), nil, true},
		{mat.NewScalar[T](5.0), nil, true},
		{mat.NewScalar[T](0.0), nil, true},
		{mat.NewScalar[T](-4.0), nil, true},
	}

	max := NewScalarMax(xs)
	assert.Equal(t, xs, max.Operands())

	y := max.Forward()
	assert.InDeltaSlice(t, []T{5.0}, y.Data(), 1.0e-6)

	max.Backward(mat.NewScalar[T](1.0))

	assert.InDeltaSlice(t, []T{1.0}, xs[1].grad.Data(), 1.0e-6)
	assert.Nil(t, xs[0].grad)
	assert.Nil(t, xs[2].grad)
	assert.Nil(t, xs[3].grad)
}
