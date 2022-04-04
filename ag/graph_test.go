// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
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
		assert.Equal(t, -1, g.maxID)
		assert.Equal(t, 0, g.curTimeStep)
		assert.Nil(t, g.nodes)
		assert.Empty(t, g.constants)
	}

	t.Run("without option", func(t *testing.T) {
		g := NewGraph[T]()
		runCommonAssertions(t, g)
		assert.NotNil(t, g.randGen)
	})

	t.Run("with WithRand option", func(t *testing.T) {
		r := rand.NewLockedRand[T](42)
		g := NewGraph[T](WithRand(r))
		runCommonAssertions(t, g)
		assert.Same(t, r, g.randGen)
	})

	t.Run("with WithRandSeed option", func(t *testing.T) {
		r := rand.NewLockedRand[T](42)
		g := NewGraph[T](WithRandSeed[T](42))
		runCommonAssertions(t, g)
		assert.NotNil(t, g.randGen)
		assert.Equal(t, r.Int(), g.randGen.Int())
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

	t.Run("it assigns the correct ID to the nodes and adds them to the graph", func(t *testing.T) {
		g := NewGraph[T]()
		a := mat.NewScalar[T](1)
		b := mat.NewScalar[T](2)
		va := g.NewVariable(a, true)
		vb := g.NewVariable(b, false)
		assert.Equal(t, 0, va.ID())
		assert.Equal(t, 1, vb.ID())
		assert.Equal(t, []Node[T]{va, vb}, g.nodes)
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
	assert.Same(t, c, g.Constant(42))
	assert.NotSame(t, c, g.Constant(43))
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
	t.Run("it resets maxID", func(t *testing.T) {
		g := NewGraph[T]()
		g.NewScalar(42)
		assert.Equal(t, 0, g.maxID)
		g.Clear(false)
		assert.Equal(t, -1, g.maxID)
	})

	t.Run("it resets curTimeStep", func(t *testing.T) {
		g := NewGraph[T]()
		g.NewScalar(42)
		g.IncTimeStep()
		assert.Equal(t, 1, g.curTimeStep)
		g.Clear(false)
		assert.Equal(t, 0, g.curTimeStep)
	})

	t.Run("it resets nodes", func(t *testing.T) {
		g := NewGraph[T]()
		g.NewScalar(42)
		assert.NotNil(t, g.nodes)
		g.Clear(false)
		assert.Nil(t, g.nodes)
	})

	t.Run("operators memory (values and grads) is released", func(t *testing.T) {
		g := NewGraph[T]()
		op := Add(
			g.NewVariable(mat.NewScalar[T](1), true),
			g.NewVariable(mat.NewScalar[T](2), true),
		)
		op.Value() // wait for the value
		Backward(op)

		assert.NotNil(t, op.Value())
		assert.NotNil(t, op.Grad())

		g.Clear(false)

		assert.Panics(t, func() { op.(*Operator[T]).Value() })
		assert.Nil(t, op.Grad())
	})

	t.Run("operators memory (values and grads) is cleared for reuse", func(t *testing.T) {
		g := NewGraph[T]()
		op := Add(
			g.NewVariable(mat.NewScalar[T](1), true),
			g.NewVariable(mat.NewScalar[T](2), true),
		)
		op.Value() // wait for the value
		Backward(op)

		assert.NotNil(t, op.Value())
		assert.NotNil(t, op.Grad())

		g.Clear(true)

		assert.Nil(t, op.Grad())
	})

	t.Run("it works on a graph without nodes", func(t *testing.T) {
		g := NewGraph[T]()
		g.Clear(false)
		assert.Equal(t, -1, g.maxID)
		assert.Equal(t, 0, g.curTimeStep)
		assert.Nil(t, g.nodes)
	})
}

func TestGraph_ClearForReuse(t *testing.T) {
	t.Run("float32", testGraphClearForReuse[float32])
	t.Run("float64", testGraphClearForReuse[float64])
}

func testGraphClearForReuse[T mat.DType](t *testing.T) {
	t.Run("operators memory (values and grads) is released", func(t *testing.T) {
		g := NewGraph[T]()
		op := Add(
			g.NewVariable(mat.NewScalar[T](1), true),
			g.NewVariable(mat.NewScalar[T](2), true),
		)
		Backward(op)

		assert.NotNil(t, op.Value())
		assert.Equal(t, T(3), op.Value().Scalar())
		assert.NotNil(t, op.Grad())

		g.Clear(true)
		g.Forward()

		assert.NotNil(t, op.Value())
		assert.Equal(t, T(3), op.Value().Scalar())
		assert.Nil(t, op.Grad())
	})

	t.Run("it works on a graph without nodes", func(t *testing.T) {
		g := NewGraph[T]()
		assert.NotPanics(t, func() { g.Clear(true) })
	})
}

func TestGraph_ZeroGrad(t *testing.T) {
	t.Run("float32", testGraphZeroGrad[float32])
	t.Run("float64", testGraphZeroGrad[float64])
}

func testGraphZeroGrad[T mat.DType](t *testing.T) {
	g := NewGraph[T]()
	v1 := g.NewVariable(mat.NewScalar[T](1), true)
	v2 := g.NewVariable(mat.NewScalar[T](2), true)
	op := Add(v1, v2)
	Backward(op)

	assert.NotNil(t, v1.Grad())
	assert.NotNil(t, v2.Grad())
	assert.NotNil(t, op.Grad())

	g.ZeroGrad()

	assert.Nil(t, v1.Grad())
	assert.Nil(t, v2.Grad())
	assert.Nil(t, op.Grad())
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
	assert.Equal(t, 0, w.id)
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
	assert.Equal(t, 0, w.id)
	assert.False(t, w.wrapGrad)
}

func TestGraph_Forward(t *testing.T) {
	t.Run("float32", testGraphForward[float32])
	t.Run("float64", testGraphForward[float64])
}

func testGraphForward[T mat.DType](t *testing.T) {
	g := NewGraph[T]()
	x1 := g.NewScalar(40)
	x2 := g.NewScalar(2)
	op := Add(x1, x2)
	assert.NotNil(t, op.Value())
	assert.Equal(t, T(42), op.Value().Scalar())
	g.Clear(true)
	ReplaceValue[T](x1, mat.NewScalar[T](60))
	ReplaceValue[T](x2, mat.NewScalar[T](9))
	g.Forward()
	assert.NotNil(t, op.Value())
	assert.Equal(t, T(69), op.Value().Scalar())
}
