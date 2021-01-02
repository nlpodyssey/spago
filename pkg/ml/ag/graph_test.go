// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewGraph(t *testing.T) {
	runCommonAssertions := func(t *testing.T, g *Graph) {
		t.Helper()
		assert.NotNil(t, g)
		assert.Equal(t, -1, g.maxID)
		assert.Equal(t, 0, g.curTimeStep)
		assert.Nil(t, g.nodes)
		assert.Empty(t, g.constants)
		assert.Equal(t, -1, g.cache.maxID)
		assert.Nil(t, g.cache.nodesByHeight)
		assert.Nil(t, g.cache.height)
	}

	t.Run("without option", func(t *testing.T) {
		g := NewGraph()
		runCommonAssertions(t, g)
		assert.NotNil(t, g.randGen)
		assert.True(t, g.incrementalForward)
		assert.Equal(t, defaultProcessingQueueSize, g.ConcurrentComputations())
	})

	t.Run("with IncrementalForward(false) option", func(t *testing.T) {
		g := NewGraph(IncrementalForward(false))
		runCommonAssertions(t, g)
		assert.NotNil(t, g.randGen)
		assert.False(t, g.incrementalForward)
		assert.Equal(t, defaultProcessingQueueSize, g.ConcurrentComputations())
	})

	t.Run("with ConcurrentComputations option", func(t *testing.T) {
		for i := 1; i < 4; i++ {
			size := i
			t.Run(fmt.Sprintf("size %d", size), func(t *testing.T) {
				g := NewGraph(ConcurrentComputations(size))
				runCommonAssertions(t, g)
				assert.NotNil(t, g.randGen)
				assert.True(t, g.incrementalForward)
				assert.Equal(t, size, g.ConcurrentComputations())
			})
		}
	})

	t.Run("with Rand option", func(t *testing.T) {
		r := rand.NewLockedRand(42)
		g := NewGraph(Rand(r))
		runCommonAssertions(t, g)
		assert.Same(t, r, g.randGen)
		assert.True(t, g.incrementalForward)
		assert.Equal(t, defaultProcessingQueueSize, g.ConcurrentComputations())
	})

	t.Run("with RandSeed option", func(t *testing.T) {
		r := rand.NewLockedRand(42)
		g := NewGraph(RandSeed(42))
		runCommonAssertions(t, g)
		assert.NotNil(t, g.randGen)
		assert.Equal(t, r.Int(), g.randGen.Int())
		assert.True(t, g.incrementalForward)
		assert.Equal(t, defaultProcessingQueueSize, g.ConcurrentComputations())
	})
}

func TestConcurrentComputations(t *testing.T) {
	t.Run("it panics if value < 1", func(t *testing.T) {
		assert.Panics(t, func() { ConcurrentComputations(0) })
	})
}

func TestGraph_NewVariable(t *testing.T) {
	t.Run("with requiresGrad true", func(t *testing.T) {
		g := NewGraph()
		s := mat.NewScalar(1)
		v := g.NewVariable(s, true)
		assert.NotNil(t, v)
		assert.Same(t, s, v.Value())
		assert.True(t, v.RequiresGrad())
	})

	t.Run("with requiresGrad false", func(t *testing.T) {
		g := NewGraph()
		s := mat.NewScalar(1)
		v := g.NewVariable(s, false)
		assert.NotNil(t, v)
		assert.Same(t, s, v.Value())
		assert.False(t, v.RequiresGrad())
	})

	t.Run("it assigns the correct ID to the nodes and adds them to the graph", func(t *testing.T) {
		g := NewGraph()
		a := mat.NewScalar(1)
		b := mat.NewScalar(2)
		va := g.NewVariable(a, true)
		vb := g.NewVariable(b, false)
		assert.Equal(t, 0, va.ID())
		assert.Equal(t, 1, vb.ID())
		assert.Equal(t, []Node{va, vb}, g.nodes)
	})
}

func TestGraph_NewScalar(t *testing.T) {
	g := NewGraph()
	s := g.NewScalar(42)
	assert.NotNil(t, s)
	assert.False(t, s.RequiresGrad())
	v := s.Value()
	assert.NotNil(t, v)
	assert.True(t, v.IsScalar())
	assert.Equal(t, mat.Float(42.0), v.Scalar())
}

func TestGraph_Constant(t *testing.T) {
	g := NewGraph()
	c := g.Constant(42)
	assert.NotNil(t, c)
	assert.False(t, c.RequiresGrad())
	v := c.Value()
	assert.NotNil(t, v)
	assert.True(t, v.IsScalar())
	assert.Equal(t, mat.Float(42.0), v.Scalar())
	assert.Same(t, c, g.Constant(42))
	assert.NotSame(t, c, g.Constant(43))
}

func TestGraph_IncTimeStep(t *testing.T) {
	g := NewGraph()
	assert.Equal(t, 0, g.TimeStep())

	g.IncTimeStep()
	assert.Equal(t, 1, g.TimeStep())

	g.IncTimeStep()
	assert.Equal(t, 2, g.TimeStep())
}

func TestNodesTimeStep(t *testing.T) {
	g := NewGraph()

	a := g.NewVariable(mat.NewScalar(1), false)
	assert.Equal(t, 0, a.TimeStep())

	g.IncTimeStep()
	b := g.NewVariable(mat.NewScalar(2), false)
	assert.Equal(t, 1, b.TimeStep())

	g.IncTimeStep()
	c := g.NewVariable(mat.NewScalar(3), false)
	assert.Equal(t, 2, c.TimeStep())
}

func TestGraph_Clear(t *testing.T) {
	t.Run("it resets maxID", func(t *testing.T) {
		g := NewGraph()
		g.NewScalar(42)
		assert.Equal(t, 0, g.maxID)
		g.Clear()
		assert.Equal(t, -1, g.maxID)
	})

	t.Run("it resets curTimeStep", func(t *testing.T) {
		g := NewGraph()
		g.NewScalar(42)
		g.IncTimeStep()
		assert.Equal(t, 1, g.curTimeStep)
		g.Clear()
		assert.Equal(t, 0, g.curTimeStep)
	})

	t.Run("it resets nodes", func(t *testing.T) {
		g := NewGraph()
		g.NewScalar(42)
		assert.NotNil(t, g.nodes)
		g.Clear()
		assert.Nil(t, g.nodes)
	})

	t.Run("it resets the cache", func(t *testing.T) {
		g := NewGraph()
		g.Add(g.NewScalar(1), g.NewScalar(2))
		g.groupNodesByHeight() // it's just a function which uses the cache

		assert.NotEqual(t, 0, g.cache.maxID)
		assert.NotNil(t, g.cache.nodesByHeight)
		assert.NotNil(t, g.cache.height)

		g.Clear()

		assert.Equal(t, -1, g.cache.maxID)
		assert.Nil(t, g.cache.nodesByHeight)
		assert.Nil(t, g.cache.height)
	})

	t.Run("operators memory (values and grads) is released", func(t *testing.T) {
		g := NewGraph()
		op := g.Add(
			g.NewVariable(mat.NewScalar(1), true),
			g.NewVariable(mat.NewScalar(2), true),
		)
		g.Backward(op)

		assert.NotNil(t, op.Value())
		assert.NotNil(t, op.Grad())

		g.Clear()

		assert.Nil(t, op.Value())
		assert.Nil(t, op.Grad())
	})

	t.Run("it works on a graph without nodes", func(t *testing.T) {
		g := NewGraph()
		g.Clear()
		assert.Equal(t, -1, g.maxID)
		assert.Equal(t, 0, g.curTimeStep)
		assert.Nil(t, g.nodes)
	})
}

func TestGraph_ClearForReuse(t *testing.T) {
	t.Run("operators memory (values and grads) is released", func(t *testing.T) {
		g := NewGraph()
		op := g.Add(
			g.NewVariable(mat.NewScalar(1), true),
			g.NewVariable(mat.NewScalar(2), true),
		)
		g.Backward(op)

		assert.NotNil(t, op.Value())
		assert.NotNil(t, op.Grad())

		g.ClearForReuse()

		assert.Nil(t, op.Value())
		assert.Nil(t, op.Grad())
	})

	t.Run("it works on a graph without nodes", func(t *testing.T) {
		g := NewGraph()
		assert.NotPanics(t, func() { g.ClearForReuse() })
	})
}

func TestGraph_ZeroGrad(t *testing.T) {
	g := NewGraph()
	v1 := g.NewVariable(mat.NewScalar(1), true)
	v2 := g.NewVariable(mat.NewScalar(2), true)
	op := g.Add(v1, v2)
	g.Backward(op)

	assert.NotNil(t, v1.Grad())
	assert.NotNil(t, v2.Grad())
	assert.NotNil(t, op.Grad())

	g.ZeroGrad()

	assert.Nil(t, v1.Grad())
	assert.Nil(t, v2.Grad())
	assert.Nil(t, op.Grad())
}

func TestGraph_NewOperator(t *testing.T) {
	t.Run("it panics if operands belong to a different Graph", func(t *testing.T) {
		g1 := NewGraph()
		g2 := NewGraph()
		x := g2.NewScalar(42)
		assert.Panics(t, func() { g1.NewOperator(fn.NewSqrt(x), x) })
	})
}

func TestGraph_NewWrap(t *testing.T) {
	s := NewGraph().NewScalar(42)
	g := NewGraph()
	g.IncTimeStep()

	result := g.NewWrap(s)
	assert.IsType(t, &wrapper{}, result)
	w := result.(*wrapper)

	assert.Same(t, s, w.GradValue)
	assert.Equal(t, 1, w.timeStep)
	assert.Same(t, g, w.graph)
	assert.Equal(t, 0, w.id)
	assert.True(t, w.wrapGrad)
}

func TestGraph_NewWrapNoGrad(t *testing.T) {
	s := NewGraph().NewScalar(42)
	g := NewGraph()
	g.IncTimeStep()

	result := g.NewWrapNoGrad(s)
	assert.IsType(t, &wrapper{}, result)
	w := result.(*wrapper)

	assert.Same(t, s, w.GradValue)
	assert.Equal(t, 1, w.timeStep)
	assert.Same(t, g, w.graph)
	assert.Equal(t, 0, w.id)
	assert.False(t, w.wrapGrad)
}

func TestGraph_Forward(t *testing.T) {
	g := NewGraph(IncrementalForward(false))
	op := g.Add(g.NewScalar(40), g.NewScalar(2))
	assert.Nil(t, op.Value())
	g.Forward()
	assert.NotNil(t, op.Value())
	assert.Equal(t, mat.Float(42.0), op.Value().Scalar())
}
