// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestNewGraph(t *testing.T) {
	t.Run("float32", testNewGraph[float32])
	t.Run("float64", testNewGraph[float64])
}

func testNewGraph[T mat.DType](t *testing.T) {
	runCommonAssertions := func(t *testing.T, g *Graph[T]) {
		t.Helper()
		assert.NotNil(t, g)
		assert.Equal(t, 0, g.curTimeStep)
	}

	t.Run("new graph", func(t *testing.T) {
		g := NewGraph[T]()
		runCommonAssertions(t, g)
	})
}

func TestGraph_NewVariable(t *testing.T) {
	t.Run("float32", testGraphNewVariable[float32])
	t.Run("float64", testGraphNewVariable[float64])
}

func testGraphNewVariable[T mat.DType](t *testing.T) {
	t.Run("with requiresGrad true", func(t *testing.T) {
		g := NewGraph[T]()
		s := mat.NewScalar[T](1)
		v := g.NewVariable(s, true)
		assert.NotNil(t, v)
		assert.Same(t, s, v.Value())
		assert.True(t, v.RequiresGrad())
	})

	t.Run("with requiresGrad false", func(t *testing.T) {
		g := NewGraph[T]()
		s := mat.NewScalar[T](1)
		v := g.NewVariable(s, false)
		assert.NotNil(t, v)
		assert.Same(t, s, v.Value())
		assert.False(t, v.RequiresGrad())
	})
}

func TestGraph_NewScalar(t *testing.T) {
	t.Run("float32", testGraphNewScalar[float32])
	t.Run("float64", testGraphNewScalar[float64])
}

func testGraphNewScalar[T mat.DType](t *testing.T) {
	g := NewGraph[T]()
	s := g.NewScalar(42)
	assert.NotNil(t, s)
	assert.False(t, s.RequiresGrad())
	v := s.Value()
	assert.NotNil(t, v)
	assert.True(t, mat.IsScalar(v))
	assert.Equal(t, T(42.0), v.Scalar())
}

func TestGraph_Constant(t *testing.T) {
	t.Run("float32", testGraphConstant[float32])
	t.Run("float64", testGraphConstant[float64])
}

func testGraphConstant[T mat.DType](t *testing.T) {
	g := NewGraph[T]()
	c := g.Constant(42)
	assert.NotNil(t, c)
	assert.False(t, c.RequiresGrad())
	v := c.Value()
	assert.NotNil(t, v)
	assert.True(t, mat.IsScalar(v))
	assert.Equal(t, T(42.0), v.Scalar())
}

func TestGraph_IncTimeStep(t *testing.T) {
	t.Run("float32", testGraphIncTimeStep[float32])
	t.Run("float64", testGraphIncTimeStep[float64])
}

func testGraphIncTimeStep[T mat.DType](t *testing.T) {
	g := NewGraph[T]()
	assert.Equal(t, 0, g.TimeStep())

	g.IncTimeStep()
	assert.Equal(t, 1, g.TimeStep())

	g.IncTimeStep()
	assert.Equal(t, 2, g.TimeStep())
}

func TestNodesTimeStep(t *testing.T) {
	t.Run("float32", testNodesTimeStep[float32])
	t.Run("float64", testNodesTimeStep[float64])
}

func testNodesTimeStep[T mat.DType](t *testing.T) {
	g := NewGraph[T]()

	a := g.NewVariable(mat.NewScalar[T](1), false)
	assert.Equal(t, 0, a.TimeStep())

	g.IncTimeStep()
	b := g.NewVariable(mat.NewScalar[T](2), false)
	assert.Equal(t, 1, b.TimeStep())

	g.IncTimeStep()
	c := g.NewVariable(mat.NewScalar[T](3), false)
	assert.Equal(t, 2, c.TimeStep())
}

func TestGraph_Clear(t *testing.T) {
	t.Run("float32", testGraphClear[float32])
	t.Run("float64", testGraphClear[float64])
}

func testGraphClear[T mat.DType](t *testing.T) {
	t.Run("it resets curTimeStep", func(t *testing.T) {
		g := NewGraph[T]()
		g.NewScalar(42)
		g.IncTimeStep()
		assert.Equal(t, 1, g.curTimeStep)
		g.Clear()
		assert.Equal(t, 0, g.curTimeStep)
	})

	t.Run("it works on a graph without nodes", func(t *testing.T) {
		g := NewGraph[T]()
		g.Clear()
		assert.Equal(t, 0, g.curTimeStep)
	})
}

func TestGraph_NewWrap(t *testing.T) {
	t.Run("float32", testGraphNewWrap[float32])
	t.Run("float64", testGraphNewWrap[float64])
}

func testGraphNewWrap[T mat.DType](t *testing.T) {
	s := NewGraph[T]().NewScalar(42)
	g := NewGraph[T]()
	g.IncTimeStep()

	result := g.NewWrap(s)
	assert.IsType(t, &Wrapper[T]{}, result)
	w := result.(*Wrapper[T])

	assert.Same(t, s, w.GradValue)
	assert.Equal(t, 1, w.timeStep)
	assert.Same(t, g, w.graph)
	assert.True(t, w.wrapGrad)
}

func TestGraph_NewWrapNoGrad(t *testing.T) {
	t.Run("float32", testGraphNewWrapNoGrad[float32])
	t.Run("float64", testGraphNewWrapNoGrad[float64])
}

func testGraphNewWrapNoGrad[T mat.DType](t *testing.T) {
	s := NewGraph[T]().NewScalar(42)
	g := NewGraph[T]()
	g.IncTimeStep()

	result := g.NewWrapNoGrad(s)
	assert.IsType(t, &Wrapper[T]{}, result)
	w := result.(*Wrapper[T])

	assert.Same(t, s, w.GradValue)
	assert.Equal(t, 1, w.timeStep)
	assert.Same(t, g, w.graph)
	assert.False(t, w.wrapGrad)
}
