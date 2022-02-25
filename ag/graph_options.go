// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/utils/processingqueue"
	"runtime"
)

// defaultProcessingQueueSize is the default size of Graph.processingQueue on a new Graph.
var defaultProcessingQueueSize = runtime.NumCPU()

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

// WithEagerExecution sets whether to compute the forward during the graph definition (default true).
// When enabled it lets you access to the Value() resulting from the computation.
// There are particular cases where you don't need intermediate values so computing the forward after
// the graph definition can be more efficient.
func WithEagerExecution[T mat.DType](value bool) GraphOption[T] {
	return func(g *Graph[T]) {
		g.eagerExecution = value
	}
}

// WithConcurrentComputations sets the maximum number of concurrent computations handled by the Graph
// for heavy tasks such as forward and backward steps.
// The value 1 corresponds to sequential execution.
func WithConcurrentComputations[T mat.DType](value int) GraphOption[T] {
	if value < 1 {
		panic("ag: WithConcurrentComputations value must be greater than zero")
	}
	return func(g *Graph[T]) {
		g.processingQueue = processingqueue.New(value)
	}
}

// WithMode sets whether the graph is being used in training or inference.
func WithMode[T mat.DType](mode ProcessingMode) GraphOption[T] {
	return func(g *Graph[T]) {
		g.mode = mode
	}
}
