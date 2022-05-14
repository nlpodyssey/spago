// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// Param is the interface for a Model parameter.
type Param[T mat.DType] interface {
	ag.Node[T]

	// Name returns the params name (can be empty string).
	Name() string
	// Type returns the params type (weights, biases, undefined).
	Type() ParamsType
	// SetRequiresGrad set whether the param requires gradient, or not.
	SetRequiresGrad(value bool)
	// ReplaceValue replaces the value of the parameter and clears the support structure.
	ReplaceValue(value mat.Matrix)
	// ApplyDelta updates the value applying the delta.
	ApplyDelta(delta mat.Matrix)
	// Payload returns the optimizer support structure (can be nil).
	Payload() *Payload[T]
	// SetPayload is a thread safe operation to set the given Payload on the
	// receiver Param.
	SetPayload(payload *Payload[T])
	// ClearPayload clears the support structure.
	ClearPayload()
}

// ParamNameSetter is implemented by any parameter value that allows the
// assignment of its name.
type ParamNameSetter interface {
	// SetName sets the parameter's name (it can be an empty string).
	SetName(string)
}

// ParamTypeSetter is implemented by any parameter value that allows the
// assignment of its type.
type ParamTypeSetter interface {
	// SetType sets the parameter's type (Weights, Biases, or Undefined).
	SetType(ParamsType)
}
