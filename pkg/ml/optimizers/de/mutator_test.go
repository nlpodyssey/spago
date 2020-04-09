// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestRandomMutator(t *testing.T) {
	population := newTestMutator()
	mutation := NewRandomMutation(6.0)
	mutation.Mutate(population)

	if !floats.EqualApprox(population.Members[0].DonorVector.Data(), []float64{0.85, -0.1, 2.1, 0.2, 2.85}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[1].DonorVector.Data(), []float64{-0.95, 1.8, 0.05, -3.3, -2.25}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[2].DonorVector.Data(), []float64{-1.7, 0.75, -2.1, 2.0, -1.1}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[3].DonorVector.Data(), []float64{-1.45, 0.85, -0.25, -2.7, -1.9}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[4].DonorVector.Data(), []float64{-2.1, 0.95, 0.45, 0.0, 2.2}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[5].DonorVector.Data(), []float64{2.95, -0.8, 1.65, -0.4, 1.1}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[6].DonorVector.Data(), []float64{-1.5, -0.9, -0.5, 0.25, 0.6}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[7].DonorVector.Data(), []float64{4.7, -2.7, 2.25, 2.75, 3.05}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[8].DonorVector.Data(), []float64{-1.55, 0.15, -1.45, 1.15, -0.65}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[9].DonorVector.Data(), []float64{-0.9, -2.1, -1.35, 4.7, 1.3}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
}

func TestDeglMutator(t *testing.T) {
	population := newTestMutator()
	mutation := NewDeglMutation(0.3, 6.0)
	mutation.Mutate(population)

	if !floats.EqualApprox(population.Members[0].DonorVector.Data(), []float64{-1.35, 0.05, 0.15, 1.075, 1.875}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[1].DonorVector.Data(), []float64{-0.975, -0.7, 0.65, -0.35, 2.3}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[2].DonorVector.Data(), []float64{1.4, -2.025, 1.05, 1.625, 2.2}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[3].DonorVector.Data(), []float64{1, -0.375, 0.575, 1.275, 1.625}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[4].DonorVector.Data(), []float64{-0.9, -1.675, 0.95, -1.4, 2.425}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[5].DonorVector.Data(), []float64{-0.85, -0.05, 0.525, -2.5, -0.4}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[6].DonorVector.Data(), []float64{-1.125, 0.725, 0.75, -3.525, -0.7}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[7].DonorVector.Data(), []float64{1.025, -2.375, 0.15, 3.175, 2.225}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[8].DonorVector.Data(), []float64{1.15, -0.3, 0.05, 0.325, 0.425}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
	if !floats.EqualApprox(population.Members[9].DonorVector.Data(), []float64{-0.275, -0.625, -0.675, 0.075, 0.175}, 0.0001) {
		t.Error("Donor vector doesn't match expected values")
	}
}

func newTestMutator() *Population {
	hyperParams := MemberHyperParams{
		MutationFactor: 0.5,
		CrossoverRate:  0.9,
		WeightFactor:   0.5,
	}

	population := NewRandomPopulation(10, 5, 6.0, rand.NewLockedRand(42), hyperParams)

	population.Members[0].TargetVector.SetData([]float64{0.0, 0.6, 0.8, 1.2, 1.6})
	population.Members[0].TrialScore = 0.3
	population.Members[0].TargetScore = 0.05
	population.Members[0].ValidationScore = 33.0

	population.Members[1].TargetVector.SetData([]float64{-0.4, 0.9, 1.8, -1.5, 2.6})
	population.Members[1].TrialScore = 0.2
	population.Members[1].TargetScore = 0.77
	population.Members[1].ValidationScore = 66.0

	population.Members[2].TargetVector.SetData([]float64{3.0, -0.8, 0.9, 2.2, 0.6})
	population.Members[2].TrialScore = 0.45
	population.Members[2].TargetScore = 0.27
	population.Members[2].ValidationScore = 52.0

	population.Members[3].TargetVector.SetData([]float64{0.5, 1.2, 0.3, -1.2, 0.1})
	population.Members[3].TrialScore = 0.37
	population.Members[3].TargetScore = 0.4
	population.Members[3].ValidationScore = 51.0

	population.Members[4].TargetVector.SetData([]float64{2.0, -3.6, 1.8, 1.2, 3.6})
	population.Members[4].TrialScore = 0.33
	population.Members[4].TargetScore = 0.28
	population.Members[4].ValidationScore = 57.0

	population.Members[5].TargetVector.SetData([]float64{-1.4, 0.2, -0.9, 0.1, -1.3})
	population.Members[5].TrialScore = 0.34
	population.Members[5].TargetScore = 0.28
	population.Members[5].ValidationScore = 57.0

	population.Members[6].TargetVector.SetData([]float64{-0.2, -0.6, 0.8, -2.1, -0.5})
	population.Members[6].TrialScore = 0.21
	population.Members[6].TargetScore = 0.25
	population.Members[6].ValidationScore = 54.0

	population.Members[7].TargetVector.SetData([]float64{-0.4, -0.7, -1.8, 5.2, -0.2})
	population.Members[7].TrialScore = 0.3
	population.Members[7].TargetScore = 0.6
	population.Members[7].ValidationScore = 58.0

	population.Members[8].TargetVector.SetData([]float64{0.6, 0.6, -0.5, -0.4, -0.9})
	population.Members[8].TrialScore = 0.33
	population.Members[8].TargetScore = 0.5
	population.Members[8].ValidationScore = 53.0

	population.Members[9].TargetVector.SetData([]float64{-0.5, -0.7, -0.3, 0.0, 0.8})
	population.Members[9].TrialScore = 0.8
	population.Members[9].TargetScore = 0.03
	population.Members[9].ValidationScore = 52.0

	return population
}
