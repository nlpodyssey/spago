// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package encoding_test

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/encoding"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGraph(t *testing.T) {
	type T = float32
	a := ag.NewVariable[T](mat.NewScalar[T](1), false)
	b := ag.NewVariable[T](mat.NewScalar[T](3), false)
	c := ag.NewVariable[T](mat.NewScalar[T](5), false)
	ag.SetTimeStep(c, 0)

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

	assert.Len(t, g.NodesByTimeStep, 2)
	require.Contains(t, g.NodesByTimeStep, -1)
	require.Contains(t, g.NodesByTimeStep, 0)

	m := g.NodesMap

	assert.ElementsMatch(t, g.NodesByTimeStep[-1], []int{m[a], m[b], m[x], m[y]})
	assert.ElementsMatch(t, g.NodesByTimeStep[0], []int{m[c], m[z]})

	expectedEdges := map[int][]int{
		m[a]: {m[x]},
		m[x]: {m[y]},
		m[b]: {m[y]},
		m[y]: {m[z]},
		m[c]: {m[z]},
	}
	assert.Equal(t, expectedEdges, g.Edges)
}
