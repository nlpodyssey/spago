// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/utils"
)

// Mutator is implemented by values that provides mutation operations.
type Mutator[T mat.DType] interface {
	Mutate(p *Population[T])
}

var _ Mutator[float32] = &RandomMutation[float32]{}

// RandomMutation implements a random mutation operation.
type RandomMutation[T mat.DType] struct {
	Bound T
}

// NewRandomMutation returns a new RandomMutation.
func NewRandomMutation[T mat.DType](bound T) *RandomMutation[T] {
	return &RandomMutation[T]{
		Bound: bound,
	}
}

// Mutate executes the mutation generating a "donor vector" for every element of the population.
// For each vector xi in the current generation, called target vector, a vector yi, called donor vector, is obtained
// as linear combination of some vectors in the population selected according to DE/rand/1 strategy, where
//   yi = clip(xa + MutationFactor * (xb − xc))
func (m *RandomMutation[T]) Mutate(p *Population[T]) {
	for i, member := range p.Members {
		extracted := rand.GetUniqueRandomInt(3, len(p.Members), func(r int) bool { return r != i })
		xc := p.Members[extracted[2]].TargetVector
		xb := p.Members[extracted[1]].TargetVector
		xa := p.Members[extracted[0]].TargetVector
		donor := xa.Add(xb.Sub(xc).ProdScalarInPlace(member.MutationFactor))
		donor.ClipInPlace(-m.Bound, +m.Bound)
		member.DonorVector = donor
	}
}

var _ Mutator[float32] = &DeglMutation[float32]{}

// DeglMutation implements Differential Evolution with Global and Local Neighborhoods mutation strategy.
//
// Reference:
//   "Design of Two-Channel Quadrature Mirror Filter Banks Using Differential Evolution with Global and Local Neighborhoods"
//   Authors: Pradipta Ghosh, Hamim Zafar, Joydeep Banerjee, Swagatam Das (2011)
//   (https://www.springerprofessional.de/en/design-of-two-channel-quadrature-mirror-filter-banks-using-diffe/3805398)
type DeglMutation[T mat.DType] struct {
	NeighborhoodRadius T
	Bound              T
}

// NewDeglMutation returns a new DeglMutation.
func NewDeglMutation[T mat.DType](NeighborhoodRadius, bound T) *DeglMutation[T] {
	return &DeglMutation[T]{
		NeighborhoodRadius: NeighborhoodRadius,
		Bound:              bound,
	}
}

// Mutate calculate the mutated vector (donor vector) as:
//    G = xi + MutationFactor (best − xi) + MutationFactor (xa − xb)
//    L = xi + MutationFactor (bestNeighbor − xi) + MutationFactor (xc − xd)
//    yi = clip(w * L + (1-w) * G)
func (m *DeglMutation[T]) Mutate(p *Population[T]) {
	windowSize := int(T(len(p.Members)) * m.NeighborhoodRadius)
	bestIndex, _ := p.FindBest(0, len(p.Members)-1, mat.Inf[T](+1), 0)
	for i, member := range p.Members {
		except := func(r int) bool { return r != i }
		extracted := rand.GetUniqueRandomInt(2, len(p.Members), except)
		neighbors := utils.GetNeighborsIndices(len(p.Members), i, windowSize)
		extractedNeighbors := rand.GetUniqueRandomIndices(2, neighbors, except)
		bestNeighborIndex, _ := p.FindBestNeighbor(i, windowSize)
		bestNeighbor := p.Members[bestNeighborIndex].TargetVector
		best := p.Members[bestIndex].TargetVector
		xi := member.TargetVector
		xb := p.Members[extracted[1]].TargetVector
		xa := p.Members[extracted[0]].TargetVector
		xd := p.Members[extractedNeighbors[1]].TargetVector
		xc := p.Members[extractedNeighbors[0]].TargetVector
		f := member.MutationFactor
		w := member.WeightFactor
		diff1 := xa.Sub(xb).ProdScalarInPlace(f)
		diff2 := xc.Sub(xd).ProdScalarInPlace(f)
		diff3 := best.Sub(xi).ProdScalarInPlace(f)
		diff4 := bestNeighbor.Sub(xi).ProdScalarInPlace(f)
		l := xi.Add(diff4).AddInPlace(diff2).ProdScalarInPlace(1.0 - w)
		g := xi.Add(diff3).AddInPlace(diff1).ProdScalarInPlace(w)
		donor := g.Add(l)
		donor.ClipInPlace(-m.Bound, +m.Bound)
		member.DonorVector = donor
	}
}
