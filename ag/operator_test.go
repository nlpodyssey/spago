// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestReleaseOperators(t *testing.T) {
	t.Run("float32", testReleaseOperators[float32])
	t.Run("float64", testReleaseOperators[float64])
}

func testReleaseOperators[T mat.DType](t *testing.T) {
	t.Run("operators memory (values and grads) is released", func(t *testing.T) {
		op := Add(
			NewVariable[T](mat.NewScalar[T](1), true),
			NewVariable[T](mat.NewScalar[T](2), true),
		)
		op.Value() // wait for the value
		Backward(op)

		assert.NotNil(t, op.Value())
		assert.NotNil(t, op.Grad())

		ReleaseGraph[T](op)

		assert.Panics(t, func() { op.(*Operator[T]).Value() })
		assert.Nil(t, op.Grad())
	})

	t.Run("operators memory (values and grads) is cleared for reuse", func(t *testing.T) {
		op := Add(
			NewVariable[T](mat.NewScalar[T](1), true),
			NewVariable[T](mat.NewScalar[T](2), true),
		)
		op.Value() // wait for the value
		Backward(op)

		assert.NotNil(t, op.Value())
		assert.NotNil(t, op.Grad())

		ReleaseGraph[T](op)

		assert.Nil(t, op.Grad())
	})
}
