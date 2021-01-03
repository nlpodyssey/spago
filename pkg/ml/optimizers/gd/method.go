// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
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
)

// MethodConfig is an empty interface implemented by the configuration structures of
// AdaGrad, Adam, RMSProp and SGD.
type MethodConfig interface{}

// Method is implemented by any optimization method.
type Method interface {
	// Label returns the enumeration-like value which identifies this gradient descent method.
	Label() int
	// Delta returns the difference between the current params and where the method wants it to be.
	Delta(param nn.Param) mat.Matrix
	// NewSupport returns a new support structure with the given dimensions.
	NewSupport(r, c int) *nn.Payload
}

// GetOrSetPayload returns the payload from param, if it already exists, otherwise
// a new payload is created, assigned to the param, and returned.
func GetOrSetPayload(param nn.Param, m Method) *nn.Payload {
	payload := param.Payload()
	switch {
	case payload == nil:
		payload := m.NewSupport(param.Value().Dims())
		param.SetPayload(payload)
		return payload
	case payload.Label == None:
		payload := m.NewSupport(param.Value().Dims())
		param.SetPayload(payload)
		return payload
	case payload.Label == m.Label():
		return payload
	default:
		panic("gd: support structure non compatible with the optimization method")
	}
}
