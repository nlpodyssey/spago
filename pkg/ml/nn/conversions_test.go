// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestToNodes(t *testing.T) {
	t.Run("generic value", func(t *testing.T) {
		assert.Panics(t, func() { ToNodes[mat.Float](42) })
	})

	t.Run("nil value", func(t *testing.T) {
		assert.Panics(t, func() { ToNodes[mat.Float](nil) })
	})

	t.Run("Node value", func(t *testing.T) {
		g := ag.NewGraph[mat.Float]()
		node := g.NewScalar(42)
		assert.Equal(t, []ag.Node[mat.Float]{node}, ToNodes[mat.Float](node))
	})

	t.Run("[]Node value", func(t *testing.T) {
		g := ag.NewGraph[mat.Float]()
		node := g.NewScalar(42)
		nodes := []ag.Node[mat.Float]{node}
		assert.Equal(t, nodes, ToNodes[mat.Float](nodes))
	})
}

func TestToNode(t *testing.T) {
	t.Run("generic value", func(t *testing.T) {
		assert.Panics(t, func() { ToNode[mat.Float](42) })
	})

	t.Run("nil value", func(t *testing.T) {
		assert.Panics(t, func() { ToNode[mat.Float](nil) })
	})

	t.Run("Node value", func(t *testing.T) {
		g := ag.NewGraph[mat.Float]()
		node := g.NewScalar(42)
		assert.Same(t, node, ToNode[mat.Float](node))
	})

	t.Run("[]Node value with one item", func(t *testing.T) {
		g := ag.NewGraph[mat.Float]()
		node := g.NewScalar(42)
		nodes := []ag.Node[mat.Float]{node}
		assert.Equal(t, node, ToNode[mat.Float](nodes))
	})

	t.Run("[]Node value with no items", func(t *testing.T) {
		var nodes []ag.Node[mat.Float]
		assert.Panics(t, func() { ToNode[mat.Float](nodes) })
	})

	t.Run("[]Node value with two items", func(t *testing.T) {
		g := ag.NewGraph[mat.Float]()
		nodes := []ag.Node[mat.Float]{g.NewScalar(1), g.NewScalar(2)}
		assert.Panics(t, func() { ToNode[mat.Float](nodes) })
	})
}
