// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestWrapper_NewWrap(t *testing.T) {
	t.Run("float32", testWrapperNewWrap[float32])
	t.Run("float64", testWrapperNewWrap[float64])
}

func testWrapperNewWrap[T mat.DType](t *testing.T) {
	s := NewScalar[T](42)

	result := NewWrap[T](s)
	assert.IsType(t, &Wrapper[T]{}, result)
	w := result.(*Wrapper[T])
	w.IncTimeStep()

	assert.Same(t, s, w.GradValue)
	assert.Equal(t, 0, w.timeStep)
	assert.True(t, w.wrapGrad)
}

func TestWrapper_NewWrapNoGrad(t *testing.T) {
	t.Run("float32", testWrapperNewWrapNoGrad[float32])
	t.Run("float64", testWrapperNewWrapNoGrad[float64])
}

func testWrapperNewWrapNoGrad[T mat.DType](t *testing.T) {
	s := NewScalar[T](42)

	result := NewWrapNoGrad[T](s)
	assert.IsType(t, &Wrapper[T]{}, result)
	w := result.(*Wrapper[T])
	w.IncTimeStep()

	assert.Same(t, s, w.GradValue)
	assert.Equal(t, 0, w.timeStep)
	assert.False(t, w.wrapGrad)
}
