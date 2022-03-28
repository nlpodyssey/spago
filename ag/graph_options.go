// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
)

// GraphOption allows to configure a new Graph with your specific needs.
type GraphOption[T mat.DType] func(*Graph[T])

// WithRand sets the generator of random numbers.
func WithRand[T mat.DType](rand *rand.LockedRand[T]) GraphOption[T] {
	return func(g *Graph[T]) {
		g.randGen = rand
	}
}

// WithRandSeed set a new generator of random numbers with the given seed.
func WithRandSeed[T mat.DType](seed uint64) GraphOption[T] {
	return func(g *Graph[T]) {
		g.randGen = rand.NewLockedRand[T](seed)
	}
}

// WithConcurrentMode sets whether to compute the forward during the graph definition
// exploiting the concurrent computation given by goroutines.
// When active, the access to node.Value() is subject to the conclusion
// of the computation of the given node, resulting in a blocking operation.
//
// This is the default mode.
func WithConcurrentMode[T mat.DType]() GraphOption[T] {
	return func(g *Graph[T]) {
		g.executionMode = Concurrent
	}
}

// WithEagerMode sets whether to compute the forward during the graph definition.
// When enabled it lets you immediately access to the node.Value() resulting from the computation.
func WithEagerMode[T mat.DType]() GraphOption[T] {
	return func(g *Graph[T]) {
		g.executionMode = Eager
	}
}

// WithDefineMode sets whether to skip computation during the graph definition.
// When enabled it lets you access to the Value() after performing a Forward.
func WithDefineMode[T mat.DType]() GraphOption[T] {
	return func(g *Graph[T]) {
		g.executionMode = Define
	}
}
