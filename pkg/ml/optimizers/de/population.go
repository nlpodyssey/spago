// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/utils"
	"golang.org/x/exp/rand"
	"math"
)

type Population struct {
	Members []*Member
}

func NewRandomPopulation(populationSize int, vectorSize int, bound float64, rndSource rand.Source, initHyperParams MemberHyperParams) *Population {
	members := make([]*Member, populationSize)
	for i := 0; i < populationSize; i++ {
		vector := mat.NewEmptyVecDense(vectorSize)
		initializers.XavierUniform(vector, 1.0, rndSource)
		vector.ClipInPlace(-bound, +bound)
		members[i] = NewMember(vector, initHyperParams)
	}
	return &Population{
		Members: members,
	}
}

func (p *Population) FindBest(lowIndex, highIndex int, upperBound float64, initArgMin int) (argMin int, minScore float64) {
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

func (p *Population) FindBestNeighbor(index, windowSize int) (argMin int, minScore float64) {
	size := len(p.Members)
	if 2*windowSize > size {
		panic("crossover: K must be less than population size")
	}
	argMin = 0
	minScore = math.Inf(1)
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
