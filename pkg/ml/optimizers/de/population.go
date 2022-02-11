// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/utils"
)

// Population represents the population.
type Population[T mat.DType] struct {
	Members []*Member[T]
}

// NewRandomPopulation returns a new Population with members initialized randomly
// according to the given configuration.
func NewRandomPopulation[T mat.DType](
	populationSize int,
	vectorSize int,
	bound T,
	rndGen *rand.LockedRand[T],
	initHyperParams MemberHyperParams[T],
) *Population[T] {
	members := make([]*Member[T], populationSize)
	for i := 0; i < populationSize; i++ {
		vector := mat.NewEmptyVecDense[T](vectorSize)
		initializers.XavierUniform[T](vector, 1.0, rndGen)
		vector.ClipInPlace(-bound, +bound)
		members[i] = NewMember[T](vector, initHyperParams)
	}
	return &Population[T]{
		Members: members,
	}
}

// FindBest finds the best member from the Population.
func (p *Population[T]) FindBest(lowIndex, highIndex int, upperBound T, initArgMin int) (argMin int, minScore T) {
	minScore = upperBound
	argMin = initArgMin
	for i := lowIndex; i <= highIndex; i++ {
		score := p.Members[i].TargetScore
		if score < minScore {
			argMin = i
			minScore = score
		}
	}
	return
}

// FindBestNeighbor finds the best neighbor member from the Population.
func (p *Population[T]) FindBestNeighbor(index, windowSize int) (argMin int, minScore T) {
	size := len(p.Members)
	if 2*windowSize > size {
		panic("crossover: K must be less than population size")
	}
	argMin = 0
	minScore = mat.Inf[T](1)
	lowIndex := index - windowSize
	highIndex := index + windowSize
	if lowIndex < 0 {
		lowIndex = size - utils.Abs(windowSize-index)
		argMin, minScore = p.FindBest(lowIndex, size-1, minScore, lowIndex)
		argMin, minScore = p.FindBest(0, highIndex-1, minScore, argMin)
	} else if highIndex > size {
		highIndex = utils.Abs(index+windowSize) - size
		argMin, minScore = p.FindBest(lowIndex, size-1, minScore, lowIndex)
		argMin, minScore = p.FindBest(0, highIndex-1, minScore, argMin)
	} else {
		argMin, minScore = p.FindBest(lowIndex, highIndex-1, minScore, lowIndex)
	}
	return
}
