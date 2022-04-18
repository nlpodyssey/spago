// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToNodes(t *testing.T) {
	t.Run("float32", testToNodes[float32])
	t.Run("float64", testToNodes[float64])
}

func testToNodes[T mat.DType](t *testing.T) {
	n1 := &dummyNode[T]{id: 1}
	n2 := &dummyNode[T]{id: 2}

	xs := []customNodeInterface[T]{n1, n2}
	ys := ToNodes[T](xs)

	require.Len(t, ys, 2)
	assert.Same(t, n1, ys[0])
	assert.Same(t, n2, ys[1])
}

type customNodeInterface[T mat.DType] interface {
	Node[T]
	Foo()
}

type dummyNode[T mat.DType] struct {
	id int
}

func (n *dummyNode[T]) Foo()                  { panic("not implemented") }
func (n *dummyNode[T]) Value() mat.Matrix[T]  { panic("not implemented") }
func (n *dummyNode[T]) Grad() mat.Matrix[T]   { panic("not implemented") }
func (n *dummyNode[_]) HasGrad() bool         { panic("not implemented") }
func (n *dummyNode[_]) RequiresGrad() bool    { panic("not implemented") }
func (n *dummyNode[T]) AccGrad(mat.Matrix[T]) { panic("not implemented") }
func (n *dummyNode[_]) ZeroGrad()             { panic("not implemented") }
func (n *dummyNode[_]) TimeStep() int         { panic("not implemented") }
func (n *dummyNode[_]) IncTimeStep()          { panic("not implemented") }
