// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
)

// Reify returns a new structure of the same type as the one in input
// in which the fields of type Node (including those from other differentiable
// submodules) are connected to a graph.
func Reify[T mat.DType, D Differentiable[T]](i D, graphOptions ...GraphOption[T]) (D, *Graph[T]) {
	g := NewGraph[T](graphOptions...)
	reified := (&graphBinder[T]{g: g}).newBoundStruct(i).(Differentiable[T]).(D)
	return reified, g
}

// ReifyWithGraph returns a new structure of the same type as the one in input
// in which the fields of type Node (including those from other differentiable
// submodules) are connected to a graph.
func ReifyWithGraph[T mat.DType, D Differentiable[T]](g *Graph[T], i D) D {
	return (&graphBinder[T]{g: g}).newBoundStruct(i).(Differentiable[T]).(D)
}
