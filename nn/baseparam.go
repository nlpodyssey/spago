// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"sync"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

var _ Param = &BaseParam{}

// BaseParam is the default implementation satisfying the Param interface.
type BaseParam struct {
	name         string
	value        mat.Matrix // store the results of a forward evaluation.
	grad         mat.Matrix
	payload      *Payload // additional data used for example by gradient-descend optimization methods
	requiresGrad bool
	// Allows thread-safe locking for operations on value.
	valueMu sync.RWMutex
	// Allows thread-safe locking for operations on grad.
	gradMu sync.RWMutex
	// Allows thread-safe locking for operations on payload.
	payloadMu sync.RWMutex
}

// NewParam returns a new param.
func NewParam(value mat.Matrix) *BaseParam {
	return &BaseParam{
		name:         "",
		value:        value,
		grad:         nil,
		requiresGrad: true,
		payload:      nil,
	}
}

// WithGrad sets whether the param requires gradients.
// It is used to specify whether a Param should be trained or not.
func (p *BaseParam) WithGrad(value bool) *BaseParam {
	p.requiresGrad = value
	return p
}

// SetName set the params name (can be empty string).
func (p *BaseParam) SetName(name string) {
	p.name = name
}

// Name returns the params name (can be empty string).
func (p *BaseParam) Name() string {
	return p.name
}

// Value returns the value of the delegate itself.
func (p *BaseParam) Value() mat.Matrix {
	p.valueMu.RLock()
	defer p.valueMu.RUnlock()
	return p.value
}

// ReplaceValue replaces the value of the parameter and clears the gradients and
// the support structure.
func (p *BaseParam) ReplaceValue(value mat.Matrix) {
	p.ClearPayload()
	p.ZeroGrad()

	p.valueMu.Lock()
	defer p.valueMu.Unlock()
	p.value = value
}

// ScalarValue returns the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (p *BaseParam) ScalarValue() float.Float {
	p.valueMu.RLock()
	defer p.valueMu.RUnlock()
	return p.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (p *BaseParam) Grad() mat.Matrix {
	p.gradMu.RLock()
	defer p.gradMu.RUnlock()
	return p.grad
}

// AccGrad accumulate the gradients
func (p *BaseParam) AccGrad(grad mat.Matrix) {
	if !p.requiresGrad {
		return
	}
	p.gradMu.Lock()
	defer p.gradMu.Unlock()
	if p.grad == nil {
		p.grad = grad.Clone()
		return
	}
	p.grad.AddInPlace(grad)
}

// HasGrad returns true if there are accumulated gradients.
func (p *BaseParam) HasGrad() bool {
	p.gradMu.RLock()
	defer p.gradMu.RUnlock()
	return p.grad != nil
}

// RequiresGrad returns true if the param requires gradients.
func (p *BaseParam) RequiresGrad() bool {
	return p.requiresGrad
}

// SetRequiresGrad is an option to specify whether a Param should be trained or not.
func (p *BaseParam) SetRequiresGrad(value bool) {
	p.requiresGrad = value
}

// ZeroGrad clears the gradients.
func (p *BaseParam) ZeroGrad() {
	if p.grad == nil {
		return
	}
	p.gradMu.Lock()
	defer p.gradMu.Unlock()
	mat.ReleaseMatrix(p.grad)
	p.grad = nil
}

// ApplyDelta updates the value applying the delta.
func (p *BaseParam) ApplyDelta(delta mat.Matrix) {
	p.valueMu.Lock()
	defer p.valueMu.Unlock()
	p.value.SubInPlace(delta)
}

// Payload returns the optimizer support structure (can be nil).
func (p *BaseParam) Payload() *Payload {
	p.payloadMu.RLock()
	defer p.payloadMu.RUnlock()
	return p.payload
}

// SetPayload is a thread safe operation to set the given Payload on the
// receiver Param.
func (p *BaseParam) SetPayload(payload *Payload) {
	p.payloadMu.Lock()
	defer p.payloadMu.Unlock()
	p.payload = payload
}

// ClearPayload clears the support structure.
func (p *BaseParam) ClearPayload() {
	if p.payload == nil {
		return
	}
	p.payloadMu.Lock()
	defer p.payloadMu.Unlock()
	p.payload.ClearData()
	p.payload = nil
}
