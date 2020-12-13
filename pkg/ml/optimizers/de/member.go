// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"golang.org/x/exp/rand"
	"math"
)

// Member represents a member of the Population.
type Member struct {
	// The hyper-params tha might change over the generations
	MemberHyperParams
	// The target vector
	TargetVector *mat.Dense
	// The donor vector
	DonorVector *mat.Dense
	// The score of the target vector obtained during the last evaluation
	TargetScore float64
	// The score of the trial vector obtained during the last evaluation
	TrialScore float64
	// The score of the target vector on the validation set
	ValidationScore float64
}

type MemberHyperParams struct {
	// Differential weight (default 0.5)
	MutationFactor float64
	// Crossover probability (default 0.9)
	CrossoverRate float64
	// Weight factor used by DEGL mutation (default 0.5)
	WeightFactor float64
}

// l = 0.1
// u = 0.9
func (a *MemberHyperParams) MutateHyperParams(l, u float64) {
	if rand.Float64() < 0.1 {
		a.MutationFactor = l + rand.Float64()*u
	}
	if rand.Float64() < 0.1 {
		a.CrossoverRate = rand.Float64()
	}
	if rand.Float64() < 0.1 {
		a.WeightFactor = l + rand.Float64()*u
	}
}

// NewMember returns a new population member.
func NewMember(vector *mat.Dense, hyperParams MemberHyperParams) *Member {
	return &Member{
		MemberHyperParams: hyperParams,
		TargetVector:      vector,
		DonorVector:       vector.ZerosLike().(*mat.Dense),
		TargetScore:       math.Inf(1),
		TrialScore:        math.Inf(1),
	}
}
