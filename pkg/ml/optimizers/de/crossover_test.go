// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestCrossover(t *testing.T) {
	population := newTestCrossover()
	crossover := NewBinomialCrossover(rand.NewLockedRand(0))
	crossover.Crossover(population)

	assert.InDeltaSlice(t,
		[]mat.Float{0.5, 1.2, 0.3, 1.2, 0.1, 0.4, -2.6, 0.3},
		population.Members[0].DonorVector.Data(),
		0.0001,
	)
	assert.InDeltaSlice(t,
		[]mat.Float{-0.4, 0.9, 1.8, 1.2, 3.6, 1.4, -2.0, -1.8},
		population.Members[1].DonorVector.Data(),
		0.0001,
	)
	assert.InDeltaSlice(t,
		[]mat.Float{3.0, 0.2, 0.9, 0.1, 0.6, 1.4, 1.5, 1.7},
		population.Members[2].DonorVector.Data(),
		0.0001,
	)
}

func newTestCrossover() *Population {
	hyperParams := MemberHyperParams{MutationFactor: 0.5, CrossoverRate: 0.9, WeightFactor: 0.5}
	population := NewRandomPopulation(3, 8, 6.0, rand.NewLockedRand(42), hyperParams)
	population.Members[0].TargetVector.SetData([]mat.Float{0.0, 0.6, 0.8, 1.2, 1.6, 2.5, -2.6, -0.5})
	population.Members[1].TargetVector.SetData([]mat.Float{-0.4, 0.9, 1.8, -1.5, 2.6, -3.5, -2.0, 0.0})
	population.Members[2].TargetVector.SetData([]mat.Float{3.0, -0.8, 0.9, 2.2, 0.6, 0.3, 0.2, 0.1})
	population.Members[0].DonorVector.SetData([]mat.Float{0.5, 1.2, 0.3, -1.2, 0.1, 0.4, -0.8, 0.3})
	population.Members[1].DonorVector.SetData([]mat.Float{2.0, -3.6, 1.8, 1.2, 3.6, 1.4, -1.6, -1.8})
	population.Members[2].DonorVector.SetData([]mat.Float{-1.4, 0.2, -0.9, 0.1, -1.3, 1.4, 1.5, 1.7})
	return population
}
