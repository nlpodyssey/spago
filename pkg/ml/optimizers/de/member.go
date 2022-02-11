// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
)

// Member represents a member of the Population.
type Member[T mat.DType] struct {
	// The hyper-params tha might change over the generations
	MemberHyperParams[T]
	// The target vector
	TargetVector mat.Matrix[T]
	// The donor vector
	DonorVector mat.Matrix[T]
	// The score of the target vector obtained during the last evaluation
	TargetScore T
	// The score of the trial vector obtained during the last evaluation
	TrialScore T
	// The score of the target vector on the validation set
	ValidationScore T
}

// MemberHyperParams contains the hyper-parameters of a Member.
type MemberHyperParams[T mat.DType] struct {
	// Differential weight (default 0.5)
	MutationFactor T
	// Crossover probability (default 0.9)
	CrossoverRate T
	// Weight factor used by DEGL mutation (default 0.5)
	WeightFactor T
}

// MutateHyperParams mutates the hyper-parameters according to l and u.
// Suggested values: l = 0.1, u = 0.9.
func (a *MemberHyperParams[T]) MutateHyperParams(l, u T) {
	if rand.Float[T]() < 0.1 {
		a.MutationFactor = l + rand.Float[T]()*u
	}
	if rand.Float[T]() < 0.1 {
		a.CrossoverRate = rand.Float[T]()
	}
	if rand.Float[T]() < 0.1 {
		a.WeightFactor = l + rand.Float[T]()*u
	}
}

// NewMember returns a new population member.
func NewMember[T mat.DType](vector mat.Matrix[T], hyperParams MemberHyperParams[T]) *Member[T] {
	return &Member[T]{
		MemberHyperParams: hyperParams,
		TargetVector:      vector,
		DonorVector:       vector.ZerosLike(),
		TargetScore:       mat.Inf[T](1),
		TrialScore:        mat.Inf[T](1),
	}
}
