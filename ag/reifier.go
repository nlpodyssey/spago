// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
)

type Reifier[T mat.DType, D Differentiable[T]] struct {
	// the differentiable instance to reify
	module D
	// mode defines whether the graph is being used in training or inference (default inference).
	mode ProcessingMode
	// randGen is the generator of random numbers
	randGen *rand.LockedRand[T]
	// eagerExecution set whether to compute the forward during the graph definition.
	eagerExecution bool
	// graph
	graph *Graph[T]
}

// NewReifier returns a new reifier to generate differentiable "typed"-structures.
func NewReifier[T mat.DType, D Differentiable[T]](d D) Reifier[T, D] {
	return Reifier[T, D]{
		module:         d,
		mode:           Inference,
		eagerExecution: true,
		randGen:        nil, // use default graph's random generator
	}
}

// WithTrainingMode causes the new graph created for reification to have the option ag.WithMode(ag.Training).
func (r Reifier[T, D]) WithTrainingMode() Reifier[T, D] {
	r.mode = Training
	return r
}

// WithInferenceMode causes the new graph created for reification to have the option ag.WithMode(ag.Inference).
func (r Reifier[T, D]) WithInferenceMode() Reifier[T, D] {
	r.mode = Inference
	return r
}

// WithMode causes the new graph created for reification to have the option ag.WithMode.
func (r Reifier[T, D]) WithMode(mode ProcessingMode) Reifier[T, D] {
	r.mode = mode
	return r
}

// WithEagerExecution causes the new graph created for reification to have the option ag.WithEagerExecution.
func (r Reifier[T, D]) WithEagerExecution(value bool) Reifier[T, D] {
	r.eagerExecution = value
	return r
}

// WithRandGen causes the new graph created for reification to have the option ag.WithRandGen.
func (r Reifier[T, D]) WithRandGen(randGen *rand.LockedRand[T]) Reifier[T, D] {
	r.randGen = randGen
	return r
}

func (r Reifier[T, D]) WithGraph(g *Graph[T]) Reifier[T, D] {
	r.graph = g
	return r
}

// New returns a new differentiable structure of the same type as the one in input
// in which the fields of type Node (including those from other differentiable
// submodules) are connected to the same graph.
func (r Reifier[T, D]) New() (D, *Graph[T]) {
	var g = r.graph
	if g == nil {
		g = NewGraph[T](
			WithMode[T](r.mode),
			WithEagerExecution[T](r.eagerExecution),
		)
		if r.randGen != nil {
			WithRand[T](r.randGen)(g)
		}
	}
	reified := (&graphBinder[T]{g: g}).newBoundStruct(r.module).(Differentiable[T]).(D)
	return reified, g
}
