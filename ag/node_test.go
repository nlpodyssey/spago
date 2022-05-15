// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/mattest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToNodes(t *testing.T) {
	t.Run("float32", testToNodes[float32])
	t.Run("float64", testToNodes[float64])
}

func testToNodes[T mat.DType](t *testing.T) {
	n1 := &dummyNode{id: 1}
	n2 := &dummyNode{id: 2}

	xs := []customNodeInterface{n1, n2}
	ys := ToNodes(xs)

	require.Len(t, ys, 2)
	assert.Same(t, n1, ys[0])
	assert.Same(t, n2, ys[1])
}

func TestCopyValue(t *testing.T) {
	t.Run("float32", testCopyValue[float32])
	t.Run("float64", testCopyValue[float64])
}

func testCopyValue[T mat.DType](t *testing.T) {
	t.Run("nil value", func(t *testing.T) {
		n := &dummyNode{value: nil}
		v := CopyValue(n)
		assert.Nil(t, v)
	})

	t.Run("matrix value", func(t *testing.T) {
		n := &dummyNode{value: mat.NewScalar[T](42)}
		v := CopyValue(n)
		mattest.RequireMatrixEquals(t, n.value, v)
		assert.NotSame(t, n.value, v)
	})
}

func TestCopyValues(t *testing.T) {
	t.Run("float32", testCopyValues[float32])
	t.Run("float64", testCopyValues[float64])
}

func testCopyValues[T mat.DType](t *testing.T) {
	nodes := []Node{
		&dummyNode{value: mat.NewScalar[T](1)},
		&dummyNode{value: nil},
		&dummyNode{value: mat.NewScalar[T](3)},
	}
	vs := CopyValues(nodes)
	require.Len(t, vs, 3)

	mattest.RequireMatrixEquals(t, nodes[0].Value(), vs[0])
	assert.NotSame(t, nodes[0].Value(), vs[0])

	assert.Nil(t, vs[1])

	mattest.RequireMatrixEquals(t, nodes[2].Value(), vs[2])
	assert.NotSame(t, nodes[2].Value(), vs[2])
}

func TestCopyGrad(t *testing.T) {
	t.Run("float32", testCopyGrad[float32])
	t.Run("float64", testCopyGrad[float64])
}

func testCopyGrad[T mat.DType](t *testing.T) {
	t.Run("nil grad", func(t *testing.T) {
		n := &dummyNode{
			grad:         nil,
			requiresGrad: true,
		}
		v := CopyGrad(n)
		assert.Nil(t, v)
	})

	t.Run("matrix grad", func(t *testing.T) {
		n := &dummyNode{
			grad:         mat.NewScalar[T](42),
			requiresGrad: true,
		}
		v := CopyGrad(n)
		mattest.RequireMatrixEquals(t, n.grad, v)
		assert.NotSame(t, n.grad, v)
	})
}

type customNodeInterface interface {
	Node
	Foo()
}

type dummyNode struct {
	id           int // just an identifier for testing and debugging
	value        mat.Matrix
	grad         mat.Matrix
	requiresGrad bool
}

func (n *dummyNode) Foo()               { panic("not implemented") }
func (n *dummyNode) Value() mat.Matrix  { return n.value }
func (n *dummyNode) Grad() mat.Matrix   { return n.grad }
func (n *dummyNode) HasGrad() bool      { return n.grad != nil }
func (n *dummyNode) RequiresGrad() bool { return n.requiresGrad }
func (n *dummyNode) AccGrad(mat.Matrix) { panic("not implemented") }
func (n *dummyNode) ZeroGrad()          { panic("not implemented") }
func (n *dummyNode) Name() string       { panic("not implemented") }
