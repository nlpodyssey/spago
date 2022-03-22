// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestScalarMax_Forward(t *testing.T) {
	t.Run("float32", testScalarMaxForward[float32])
	t.Run("float64", testScalarMaxForward[float64])
}

func testScalarMaxForward[T mat.DType](t *testing.T) {
	xs := []*variable[T]{
		{mat.NewScalar[T](2.0), nil, true},
		{mat.NewScalar[T](5.0), nil, true},
		{mat.NewScalar[T](0.0), nil, true},
		{mat.NewScalar[T](-4.0), nil, true},
	}

	max := NewScalarMax[T](xs)
	y := max.Forward()
	assert.InDeltaSlice(t, []T{5.0}, y.Data(), 1.0e-6)

	max.Backward(mat.NewScalar[T](1.0))

	assert.InDeltaSlice(t, []T{0.0}, xs[0].grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{1.0}, xs[1].grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.0}, xs[2].grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.0}, xs[3].grad.Data(), 1.0e-6)
}
