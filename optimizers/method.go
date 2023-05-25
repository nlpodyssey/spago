// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimizers

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

const (
	// None represents the absence of a specific gradient descent optimization method.
	None int = iota
	// SGD represents the SGD gradient descent optimization method.
	SGD
	// AdaGrad represents the AdaGrad gradient descent optimization method.
	AdaGrad
	// Adam represents the Adam gradient descent optimization method.
	Adam
	// RAdam represents the RAdam gradient descent optimization method.
	RAdam
	// RMSProp represents the RMSProp gradient descent optimization method.
	RMSProp
	// Lamb represents the Lamb gradient descent optimization method.
	Lamb
)

// StrategyConfig is an empty interface implemented by the configuration structures of
// AdaGrad, Adam, RMSProp and SGD.
type StrategyConfig any

// Strategy is implemented by any optimization strategy.
type Strategy interface {
	// CalcDelta returns the difference between the current params and where the strategy wants it to be.
	CalcDelta(param *nn.Param) mat.Matrix
	// NewState returns a new support structure with the given dimensions.
	NewState(shape ...int) any
}
