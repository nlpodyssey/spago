// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"testing"
)

func TestNewRandomPopulation(t *testing.T) {
	population := newTestPopulation()

	if len(population.Members) != 20 {
		t.Error("The population size doesn't match the expected value")
	}

	for _, member := range population.Members {
		if !(member.TargetVector.Size() == 50 &&
			member.DonorVector.Size() == member.TargetVector.Size() &&
			member.DonorVector.Rows() == member.TargetVector.Rows() &&
			member.DonorVector.Columns() == member.TargetVector.Columns()) {
			t.Error("A member of the population doesn't match the expected vectors dimension")
			break
		}
	}
}

func TestFindBest(t *testing.T) {
	population := newTestPopulation()

	argMin, score := population.FindBest(0, 19, 0.3, 0)
	if argMin != 9 && score != 0.03 {
		t.Error("The result doesn't match the expected values")
	}

	argMin, score = population.FindBest(0, 19, 0.04, 0)
	if argMin != 9 && score != 0.03 {
		t.Error("The result doesn't match the expected values")
	}

	argMin, score = population.FindBest(0, 8, 0.3, 3)
	if argMin != 0 && score != 0.05 {
		t.Error("The result doesn't match the expected values")
	}

	argMin, score = population.FindBest(0, 8, 0.07, 3)
	if argMin != 0 && score != 0.05 {
		t.Error("The result doesn't match the expected values")
	}
}

func TestFindBestNeighbor(t *testing.T) {
	population := newTestPopulation()

	argMin, score := population.FindBestNeighbor(0, 5)
	if argMin != 0 && score != 0.05 {
		t.Error("The result doesn't match the expected values")
	}

	argMin, score = population.FindBestNeighbor(19, 5)
	if argMin != 0 && score != 0.05 {
		t.Error("The result doesn't match the expected values")
	}

	argMin, score = population.FindBestNeighbor(2, 6)
	if argMin != 0 && score != 0.05 {
		t.Error("The result doesn't match the expected values")
	}

	argMin, score = population.FindBestNeighbor(19, 6)
	if argMin != 0 && score != 0.05 {
		t.Error("The result doesn't match the expected values")
	}

	argMin, score = population.FindBestNeighbor(13, 6)
	if argMin != 9 && score != 0.03 {
		t.Error("The result doesn't match the expected values")
	}

	argMin, score = population.FindBestNeighbor(4, 3)
	if argMin != 6 && score != 0.25 {
		t.Error("The result doesn't match the expected values")
	}
}

func newTestPopulation() *Population {
	population := NewRandomPopulation(20, 50, 6.0, rand.NewLockedRand(42), MemberHyperParams{
		MutationFactor: 0.5,
		CrossoverRate:  0.9,
		WeightFactor:   0.5,
	})
	population.Members[0].TrialScore = 0.3
	population.Members[0].TargetScore = 0.05
	population.Members[0].ValidationScore = 33.0
	population.Members[1].TrialScore = 0.2
	population.Members[1].TargetScore = 0.77
	population.Members[1].ValidationScore = 66.0
	population.Members[2].TrialScore = 0.45
	population.Members[2].TargetScore = 0.27
	population.Members[2].ValidationScore = 52.0
	population.Members[3].TrialScore = 0.37
	population.Members[3].TargetScore = 0.4
	population.Members[3].ValidationScore = 51.0
	population.Members[4].TrialScore = 0.33
	population.Members[4].TargetScore = 0.28
	population.Members[4].ValidationScore = 57.0
	population.Members[5].TrialScore = 0.34
	population.Members[5].TargetScore = 0.28
	population.Members[5].ValidationScore = 57.0
	population.Members[6].TrialScore = 0.21
	population.Members[6].TargetScore = 0.25
	population.Members[6].ValidationScore = 54.0
	population.Members[7].TrialScore = 0.3
	population.Members[7].TargetScore = 0.6
	population.Members[7].ValidationScore = 58.0
	population.Members[8].TrialScore = 0.33
	population.Members[8].TargetScore = 0.5
	population.Members[8].ValidationScore = 53.0
	population.Members[9].TrialScore = 0.8
	population.Members[9].TargetScore = 0.03
	population.Members[9].ValidationScore = 52.0
	population.Members[10].TrialScore = 0.35
	population.Members[10].TargetScore = 0.1
	population.Members[10].ValidationScore = 34.0
	population.Members[11].TrialScore = 0.23
	population.Members[11].TargetScore = 0.23
	population.Members[11].ValidationScore = 53.4
	population.Members[12].TrialScore = 0.2
	population.Members[12].TargetScore = 0.7
	population.Members[12].ValidationScore = 56.0
	population.Members[13].TrialScore = 0.4
	population.Members[13].TargetScore = 0.5
	population.Members[13].ValidationScore = 53.5
	population.Members[14].TrialScore = 0.33
	population.Members[14].TargetScore = 0.23
	population.Members[14].ValidationScore = 54.0
	population.Members[15].TrialScore = 0.5
	population.Members[15].TargetScore = 0.28
	population.Members[15].ValidationScore = 45.0
	population.Members[16].TrialScore = 0.8
	population.Members[16].TargetScore = 0.6
	population.Members[16].ValidationScore = 57.0
	population.Members[17].TrialScore = 0.3
	population.Members[17].TargetScore = 0.1
	population.Members[17].ValidationScore = 75.0
	population.Members[18].TrialScore = 0.2
	population.Members[18].TargetScore = 0.9
	population.Members[18].ValidationScore = 52.0
	population.Members[19].TrialScore = 0.34
	population.Members[19].TargetScore = 0.24
	population.Members[19].ValidationScore = 62.0
	return population
}
