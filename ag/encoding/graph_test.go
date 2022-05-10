// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package encoding_test

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/ag/encoding"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewGraph(t *testing.T) {
	t.Run("float32", testNewGraph[float32])
	t.Run("float64", testNewGraph[float64])
}

func testNewGraph[T mat.DType](t *testing.T) {
	a := ag.NewVariable[T](mat.NewScalar[T](1), false)
	b := ag.NewVariable[T](mat.NewScalar[T](3), false)
	c := ag.NewVariable[T](mat.NewScalar[T](5), false)

	x := ag.Add(a, a)
	y := ag.Add(x, b)
	z := ag.Add(y, c)

	g := encoding.NewGraph(z)

	assert.ElementsMatch(t, []ag.Node[T]{a, b, c, x, y, z}, g.NodesList)

	expectedMap := make(map[ag.Node[T]]int, len(g.NodesList))
	for i, n := range g.NodesList {
		expectedMap[n] = i
	}
	assert.Equal(t, expectedMap, g.NodesMap)

	m := g.NodesMap
	expectedEdges := map[int][]int{
		m[a]: {m[x]},
		m[x]: {m[y]},
		m[b]: {m[y]},
		m[y]: {m[z]},
		m[c]: {m[z]},
	}
	assert.Equal(t, expectedEdges, g.Edges)

	assert.Nil(t, g.TimeStepHandler)
}

func TestGraph_WithTimeSteps(t *testing.T) {
	t.Run("float32", testGraphWithTimeSteps[float32])
	t.Run("float64", testGraphWithTimeSteps[float64])
}

func testGraphWithTimeSteps[T mat.DType](t *testing.T) {
	a := ag.NewVariable[T](mat.NewScalar[T](1), false)
	tsh := ag.NewTimeStepHandler()

	g := encoding.NewGraph(a)
	assert.Nil(t, g.TimeStepHandler)

	g2 := g.WithTimeSteps(tsh)
	assert.Same(t, tsh, g.TimeStepHandler)
	assert.Same(t, g, g2)
}

func TestGraph_NodesByTimeStep(t *testing.T) {
	t.Run("float32", testGraphNodesByTimeStep[float32])
	t.Run("float64", testGraphNodesByTimeStep[float64])
}

func testGraphNodesByTimeStep[T mat.DType](t *testing.T) {
	t.Run("without time step handler", func(t *testing.T) {
		a := ag.NewVariable[T](mat.NewScalar[T](1), false)
		b := ag.NewVariable[T](mat.NewScalar[T](3), false)
		c := ag.NewVariable[T](mat.NewScalar[T](5), false)

		x := ag.Add(a, a)
		y := ag.Add(x, b)
		z := ag.Add(y, c)

		g := encoding.NewGraph(z)

		nodesByTimeStep := g.NodesByTimeStep()
		assert.Equal(t, map[int][]int{-1: {0, 1, 2, 3, 4, 5}}, nodesByTimeStep)
	})

	t.Run("with a time step handler", func(t *testing.T) {
		tsh := ag.NewTimeStepHandler()

		a := ag.NewVariable[T](mat.NewScalar[T](1), false)
		b := ag.NewVariable[T](mat.NewScalar[T](3), false)
		c := ag.NewVariable[T](mat.NewScalar[T](5), false)

		x := ag.Add(a, a)

		tsh.IncTimeStep()
		y := ag.Add(x, b)

		tsh.IncTimeStep()
		z := ag.Add(y, c)

		g := encoding.NewGraph(z).WithTimeSteps(tsh)

		nodesByTimeStep := g.NodesByTimeStep()

		assert.Len(t, nodesByTimeStep, 3)
		require.Contains(t, nodesByTimeStep, 0)
		require.Contains(t, nodesByTimeStep, 1)
		require.Contains(t, nodesByTimeStep, 2)

		m := g.NodesMap
		assert.ElementsMatch(t, nodesByTimeStep[0], []int{m[a], m[b], m[c], m[x]})
		assert.ElementsMatch(t, nodesByTimeStep[1], []int{m[y]})
		assert.ElementsMatch(t, nodesByTimeStep[2], []int{m[z]})
	})
}
