// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestToNodes(t *testing.T) {
	t.Run("float32", testToNodes[float32])
	t.Run("float64", testToNodes[float64])
}

func testToNodes[T mat.DType](t *testing.T) {
	t.Run("generic value", func(t *testing.T) {
		assert.Panics(t, func() { ToNodes[T](42) })
	})

	t.Run("nil value", func(t *testing.T) {
		assert.Panics(t, func() { ToNodes[T](nil) })
	})

	t.Run("Node value", func(t *testing.T) {
		g := ag.NewGraph[T]()
		node := g.NewScalar(42)
		assert.Equal(t, []ag.Node[T]{node}, ToNodes[T](node))
	})

	t.Run("[]Node value", func(t *testing.T) {
		g := ag.NewGraph[T]()
		node := g.NewScalar(42)
		nodes := []ag.Node[T]{node}
		assert.Equal(t, nodes, ToNodes[T](nodes))
	})
}

func TestToNode(t *testing.T) {
	t.Run("float32", testToNode[float32])
	t.Run("float64", testToNode[float64])
}

func testToNode[T mat.DType](t *testing.T) {
	t.Run("generic value", func(t *testing.T) {
		assert.Panics(t, func() { ToNode[T](42) })
	})

	t.Run("nil value", func(t *testing.T) {
		assert.Panics(t, func() { ToNode[T](nil) })
	})

	t.Run("Node value", func(t *testing.T) {
		g := ag.NewGraph[T]()
		node := g.NewScalar(42)
		assert.Same(t, node, ToNode[T](node))
	})

	t.Run("[]Node value with one item", func(t *testing.T) {
		g := ag.NewGraph[T]()
		node := g.NewScalar(42)
		nodes := []ag.Node[T]{node}
		assert.Equal(t, node, ToNode[T](nodes))
	})

	t.Run("[]Node value with no items", func(t *testing.T) {
		var nodes []ag.Node[T]
		assert.Panics(t, func() { ToNode[T](nodes) })
	})

	t.Run("[]Node value with two items", func(t *testing.T) {
		g := ag.NewGraph[T]()
		nodes := []ag.Node[T]{g.NewScalar(1), g.NewScalar(2)}
		assert.Panics(t, func() { ToNode[T](nodes) })
	})
}
