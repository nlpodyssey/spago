// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
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
		assert.Equal(t, 0, g.cache.maxID)
		assert.Nil(t, g.cache.nodesByHeight)
		assert.Nil(t, g.cache.height)
	}

	t.Run("without option", func(t *testing.T) {
		g := NewGraph()
		runCommonAssertions(t, g)
		assert.NotNil(t, g.randGen)
		assert.True(t, g.incrementalForward)
		assert.Equal(t, 1, g.ConcurrentComputations())
	})

	t.Run("with IncrementalForward(false) option", func(t *testing.T) {
		g := NewGraph(IncrementalForward(false))
		runCommonAssertions(t, g)
		assert.NotNil(t, g.randGen)
		assert.False(t, g.incrementalForward)
		assert.Equal(t, 1, g.ConcurrentComputations())
	})

	t.Run("with ConcurrentComputations(2) option", func(t *testing.T) {
		g := NewGraph(ConcurrentComputations(2))
		runCommonAssertions(t, g)
		assert.NotNil(t, g.randGen)
		assert.True(t, g.incrementalForward)
		assert.Equal(t, 2, g.ConcurrentComputations())
	})

	t.Run("with Rand option", func(t *testing.T) {
		r := rand.NewLockedRand(42)
		g := NewGraph(Rand(r))
		runCommonAssertions(t, g)
		assert.Same(t, r, g.randGen)
		assert.True(t, g.incrementalForward)
		assert.Equal(t, 1, g.ConcurrentComputations())
	})

	t.Run("with RandSeed option", func(t *testing.T) {
		r := rand.NewLockedRand(42)
		g := NewGraph(RandSeed(42))
		runCommonAssertions(t, g)
		assert.NotNil(t, g.randGen)
		assert.Equal(t, r.Int(), g.randGen.Int())
		assert.True(t, g.incrementalForward)
		assert.Equal(t, 1, g.ConcurrentComputations())
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
	assert.Equal(t, 42.0, v.Scalar())
}

func TestGraph_Constant(t *testing.T) {
	g := NewGraph()
	c := g.Constant(42)
	assert.NotNil(t, c)
	assert.False(t, c.RequiresGrad())
	v := c.Value()
	assert.NotNil(t, v)
	assert.True(t, v.IsScalar())
	assert.Equal(t, 42.0, v.Scalar())
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
