// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
)

// Member represents a member of the Population.
type Member struct {
	// The hyper-params tha might change over the generations
	MemberHyperParams
	// The target vector
	TargetVector mat.Matrix
	// The donor vector
	DonorVector mat.Matrix
	// The score of the target vector obtained during the last evaluation
	TargetScore mat.Float
	// The score of the trial vector obtained during the last evaluation
	TrialScore mat.Float
	// The score of the target vector on the validation set
	ValidationScore mat.Float
}

// MemberHyperParams contains the hyper-parameters of a Member.
type MemberHyperParams struct {
	// Differential weight (default 0.5)
	MutationFactor mat.Float
	// Crossover probability (default 0.9)
	CrossoverRate mat.Float
	// Weight factor used by DEGL mutation (default 0.5)
	WeightFactor mat.Float
}

// MutateHyperParams mutates the hyper-parameters according to l and u.
// Suggested values: l = 0.1, u = 0.9.
func (a *MemberHyperParams) MutateHyperParams(l, u mat.Float) {
	if rand.Float() < 0.1 {
		a.MutationFactor = l + rand.Float()*u
	}
	if rand.Float() < 0.1 {
		a.CrossoverRate = rand.Float()
	}
	if rand.Float() < 0.1 {
		a.WeightFactor = l + rand.Float()*u
	}
}

// NewMember returns a new population member.
func NewMember(vector mat.Matrix, hyperParams MemberHyperParams) *Member {
	return &Member{
		MemberHyperParams: hyperParams,
		TargetVector:      vector,
		DonorVector:       vector.ZerosLike(),
		TargetScore:       mat.Inf(1),
		TrialScore:        mat.Inf(1),
	}
}
