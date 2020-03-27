// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

type MethodName int

const (
	None MethodName = iota
	SGD
	AdaGrad
	Adam
	RMSProp
)

// Support contains the support data for the optimization method
type Support struct {
	Name MethodName
	Data []mat.Matrix
}

// NewEmptySupport returns an empty support structure, not connected to any optimization method.
func NewEmptySupport() *Support {
	return &Support{
		Name: None,
		Data: make([]mat.Matrix, 0),
	}
}

type Optimizable interface {
	ag.GradValue
	// ApplyDelta updates the value of the underlying storage applying the delta.
	ApplyDelta(delta mat.Matrix)
	// Support returns the optimizer support structure (can be nil).
	Support() *Support
	// SetSupport sets the optimizer support structure. Use ClearSupport() to set a nil support.
	SetSupport(supp *Support)
	// GetOrSetSupport gets the current support structure or set a new one.
	GetOrSetSupport(m OptimizationMethod) *Support
	// ClearSupport clears the support structure.
	ClearSupport()
}

// Optimization OptimizationMethod
type OptimizationMethod interface {
	Name() MethodName
	// Delta returns the difference between the current params and where the method wants it to be.
	Delta(param Optimizable) mat.Matrix
	// NewSupport returns a new support structure with the given dimensions
	NewSupport(r, c int) *Support
}
