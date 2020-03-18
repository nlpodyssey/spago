// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"golang.org/x/exp/rand"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/initializers"
)

type Population struct {
	Members []*Member
}

func NewRandomPopulation(n int, vectorSize int, bound float64, rndSource rand.Source, initHyperParams MemberHyperParams) *Population {
	members := make([]*Member, n)
	for i := 0; i < n; i++ {
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
