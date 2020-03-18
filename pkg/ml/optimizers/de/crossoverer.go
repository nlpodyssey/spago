// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"golang.org/x/exp/rand"
	"math"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/initializers"
)

type Crossoverer interface {
	Crossover(p *Population)
}

var _ Crossoverer = &BinomialCrossover{}

type BinomialCrossover struct {
	source rand.Source
}

func NewBinomialCrossover(source rand.Source) *BinomialCrossover {
	return &BinomialCrossover{source: source}
}

func (c *BinomialCrossover) Crossover(p *Population) {
	randomVector := mat.NewEmptyVecDense(p.Members[0].DonorVector.Size())
	initializers.Uniform(randomVector, -1.0, +1.0, c.source)
	for _, member := range p.Members {
		size := member.DonorVector.Size()
		rn := rand.New(rand.NewSource(0))
		k := rn.Intn(size)
		for i := 0; i < size; i++ {
			if math.Abs(randomVector.At(i, 0)) > member.CrossoverRate || i == k { // Fixed range trick
				member.DonorVector.Set(member.TargetVector.At(i, 0), i, 0)
			}
		}
	}
}
