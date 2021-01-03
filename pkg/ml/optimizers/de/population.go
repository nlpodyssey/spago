// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/utils"
)

// Population represents the population.
type Population struct {
	Members []*Member
}

// NewRandomPopulation returns a new Population with members initialized randomly
// according to the given configuration.
func NewRandomPopulation(populationSize int, vectorSize int, bound mat.Float, rndGen *rand.LockedRand, initHyperParams MemberHyperParams) *Population {
	members := make([]*Member, populationSize)
	for i := 0; i < populationSize; i++ {
		vector := mat.NewEmptyVecDense(vectorSize)
		initializers.XavierUniform(vector, 1.0, rndGen)
		vector.ClipInPlace(-bound, +bound)
		members[i] = NewMember(vector, initHyperParams)
	}
	return &Population{
		Members: members,
	}
}

// FindBest finds the best member from the Population.
func (p *Population) FindBest(lowIndex, highIndex int, upperBound mat.Float, initArgMin int) (argMin int, minScore mat.Float) {
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
func (p *Population) FindBestNeighbor(index, windowSize int) (argMin int, minScore mat.Float) {
	size := len(p.Members)
	if 2*windowSize > size {
		panic("crossover: K must be less than population size")
	}
	argMin = 0
	minScore = mat.Inf(1)
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
