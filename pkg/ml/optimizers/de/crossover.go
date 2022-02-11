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
type Crossover[T mat.DType] interface {
	Crossover(p *Population[T])
}

var _ Crossover[float32] = &BinomialCrossover[float32]{}

// BinomialCrossover implements a binomial crossover operation.
type BinomialCrossover[T mat.DType] struct {
	rndGen *rand.LockedRand[T]
}

// NewBinomialCrossover returns a new BinomialCrossover.
func NewBinomialCrossover[T mat.DType](rndGen *rand.LockedRand[T]) *BinomialCrossover[T] {
	return &BinomialCrossover[T]{rndGen: rndGen}
}

// Crossover performs the crossover operation over the p population.
func (c *BinomialCrossover[T]) Crossover(p *Population[T]) {
	seed := rand.NewLockedRand[T](0)
	for _, member := range p.Members {
		randomVector := mat.NewEmptyVecDense[T](p.Members[0].DonorVector.Size())
		initializers.Uniform[T](randomVector, -1.0, +1.0, c.rndGen)
		size := member.DonorVector.Size()
		rn := rand.NewLockedRand[T](seed.Uint64n(100))
		k := rn.Intn(size)
		for i := 0; i < size; i++ {
			if mat.Abs(randomVector.At(i, 0)) > member.CrossoverRate || i == k { // Fixed range trick
				member.DonorVector.SetVec(i, member.TargetVector.AtVec(i))
			}
		}
	}
}
