// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestReleaseGraph(t *testing.T) {
	t.Run("float32", testReleaseGraph[float32])
	t.Run("float64", testReleaseGraph[float64])
}

func testReleaseGraph[T mat.DType](t *testing.T) {
	t.Run("values and grads are released", func(t *testing.T) {
		op := Add(
			NewVariable(mat.NewScalar[T](1), true),
			NewVariable(mat.NewScalar[T](2), true),
		)

		op.Value() // wait for the value
		Backward(op)

		assert.NotNil(t, op.Value())
		assert.NotNil(t, op.Grad())

		ReleaseGraph(op)

		assert.Panics(t, func() { op.Value() })
		assert.Nil(t, op.Grad())
	})

	t.Run("multiple occurrences of the same operator in a graph", func(t *testing.T) {
		op1 := Add(
			NewVariable(mat.NewScalar[T](1), true),
			NewVariable(mat.NewScalar[T](2), true),
		)
		op2 := Add(op1, op1)
		op2.Value() // wait for the value
		Backward(op2)

		assert.NotNil(t, op1.Value())
		assert.NotNil(t, op2.Value())

		assert.NotNil(t, op1.Grad())
		assert.NotNil(t, op2.Grad())

		ReleaseGraph(op2)

		assert.Panics(t, func() { op1.Value() })
		assert.Panics(t, func() { op2.Value() })

		assert.Nil(t, op1.Grad())
		assert.Nil(t, op2.Grad())
	})
}
