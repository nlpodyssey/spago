// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
)

// Crossover is implemented by values that provides crossover operations.
type Crossover interface {
	Crossover(p *Population)
}

var _ Crossover = &BinomialCrossover{}

// BinomialCrossover implements a binomial crossover operation.
type BinomialCrossover struct {
	rndGen *rand.LockedRand[mat.Float]
}

// NewBinomialCrossover returns a new BinomialCrossover.
func NewBinomialCrossover(rndGen *rand.LockedRand[mat.Float]) *BinomialCrossover {
	return &BinomialCrossover{rndGen: rndGen}
}

// Crossover performs the crossover operation over the p population.
func (c *BinomialCrossover) Crossover(p *Population) {
	seed := rand.NewLockedRand[mat.Float](0)
	for _, member := range p.Members {
		randomVector := mat.NewEmptyVecDense[mat.Float](p.Members[0].DonorVector.Size())
		initializers.Uniform(randomVector, -1.0, +1.0, c.rndGen)
		size := member.DonorVector.Size()
		rn := rand.NewLockedRand[mat.Float](seed.Uint64n(100))
		k := rn.Intn(size)
		for i := 0; i < size; i++ {
			if mat.Abs(randomVector.At(i, 0)) > member.CrossoverRate || i == k { // Fixed range trick
				member.DonorVector.SetVec(i, member.TargetVector.AtVec(i))
			}
		}
	}
}
