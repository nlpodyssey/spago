// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimizer

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
	// Label returns the enumeration-like value which identifies this gradient descent strategy.
	Label() int
	// CalcDelta returns the difference between the current params and where the strategy wants it to be.
	CalcDelta(param *nn.Param) mat.Matrix
	// NewPayload returns a new support structure with the given dimensions.
	NewPayload(r, c int) *nn.OptimizerPayload
}

// GetOrSetPayload returns the payload from param, if it already exists, otherwise
// a new payload is created, assigned to the param, and returned.
func GetOrSetPayload(param *nn.Param, m Strategy) *nn.OptimizerPayload {
	payload := param.Payload()
	switch {
	case payload == nil:
		payload := m.NewPayload(param.Value().Dims())
		param.SetPayload(payload)
		return payload
	case payload.Label == None:
		payload := m.NewPayload(param.Value().Dims())
		param.SetPayload(payload)
		return payload
	case payload.Label == m.Label():
		return payload
	default:
		panic("gd: support structure non compatible with the optimization strategy")
	}
}
