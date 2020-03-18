// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/mat/rnd"
)

type Mutator interface {
	Mutate(p *Population)
}

var _ Mutator = &RandomMutation{}

type RandomMutation struct {
	Bound float64
}

func NewRandomMutation(bound float64) *RandomMutation {
	return &RandomMutation{
		Bound: bound,
	}
}

// Mutate executes the mutation generating a "donor vector" for every element of the population.
// For each vector xi in the current generation, called target vector, a vector yi, called donor vector, is obtained
// as linear combination of some vectors in the population selected according to DE/rand/1 strategy, where
//   yi = clip(xa + MutationFactor * (xb − xc))
func (m *RandomMutation) Mutate(p *Population) {
	for i, member := range p.Members {
		extracted := rnd.GetUniqueRandomInt(3, len(p.Members), func(r int) bool { return r != i })
		xa := p.Members[extracted[0]].TargetVector
		xb := p.Members[extracted[1]].TargetVector
		xc := p.Members[extracted[2]].TargetVector
		donor := xa.Add(xb.Sub(xc).ProdScalarInPlace(member.MutationFactor))
		donor.ClipInPlace(-m.Bound, +m.Bound)
		member.DonorVector = donor.(*mat.Dense)
	}
}
