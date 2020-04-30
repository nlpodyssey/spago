// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

type Optimizable interface {
	ag.GradValue
	// ApplyDelta updates the value of the underlying storage applying the delta.
	ApplyDelta(delta mat.Matrix)
	// Support returns the optimizer support structure (can be nil).
	Support() *Support
	// SetSupport sets the optimizer support structure. Use ClearSupport() to set a nil support.
	SetSupport(supp *Support)
	// GetOrSetSupport gets the current support structure or set a new one.
	GetOrSetSupport(m Method) *Support
	// ClearSupport clears the support structure.
	ClearSupport()
}
