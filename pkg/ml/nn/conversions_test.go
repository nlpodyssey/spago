// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestToNodes(t *testing.T) {
	t.Run("generic value", func(t *testing.T) {
		assert.Panics(t, func() { ToNodes(42) })
	})

	t.Run("nil value", func(t *testing.T) {
		assert.Panics(t, func() { ToNodes(nil) })
	})

	t.Run("Node value", func(t *testing.T) {
		g := ag.NewGraph()
		node := g.NewScalar(42)
		assert.Equal(t, []ag.Node{node}, ToNodes(node))
	})

	t.Run("[]Node value", func(t *testing.T) {
		g := ag.NewGraph()
		node := g.NewScalar(42)
		nodes := []ag.Node{node}
		assert.Equal(t, nodes, ToNodes(nodes))
	})
}

func TestToNode(t *testing.T) {
	t.Run("generic value", func(t *testing.T) {
		assert.Panics(t, func() { ToNode(42) })
	})

	t.Run("nil value", func(t *testing.T) {
		assert.Panics(t, func() { ToNode(nil) })
	})

	t.Run("Node value", func(t *testing.T) {
		g := ag.NewGraph()
		node := g.NewScalar(42)
		assert.Same(t, node, ToNode(node))
	})

	t.Run("[]Node value with one item", func(t *testing.T) {
		g := ag.NewGraph()
		node := g.NewScalar(42)
		nodes := []ag.Node{node}
		assert.Equal(t, node, ToNode(nodes))
	})

	t.Run("[]Node value with no items", func(t *testing.T) {
		var nodes []ag.Node
		assert.Panics(t, func() { ToNode(nodes) })
	})

	t.Run("[]Node value with two items", func(t *testing.T) {
		g := ag.NewGraph()
		nodes := []ag.Node{g.NewScalar(1), g.NewScalar(2)}
		assert.Panics(t, func() { ToNode(nodes) })
	})
}
