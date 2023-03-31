// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

var _ Param = &BaseParam{}

// BaseParam is the default implementation satisfying the Param interface.
type BaseParam struct {
	mat.Matrix
	valueMu   sync.RWMutex
	payload   *Payload
	payloadMu sync.RWMutex
}

// NewParam returns a new param.
func NewParam(value mat.Matrix) *BaseParam {
	if value == nil {
		panic("nn: cannot create a new param with a nil value")
	}
	p := &BaseParam{
		Matrix:  value.Clone(),
		payload: nil,
	}
	p.SetRequiresGrad(true)
	return p
}

// WithGrad sets whether the param requires gradients.
// It is used to specify whether a Param should be trained or not.
func (p *BaseParam) WithGrad(value bool) *BaseParam {
	p.SetRequiresGrad(value)
	return p
}

// Value returns the value of the delegate itself.
func (p *BaseParam) Value() mat.Matrix {
	p.valueMu.RLock()
	defer p.valueMu.RUnlock()
	return p.Matrix
}

// ReplaceValue replaces the value of the parameter.
// It also clears the gradients and the payload.
func (p *BaseParam) ReplaceValue(value mat.Matrix) {
	p.ClearPayload()
	p.ZeroGrad()

	p.valueMu.Lock()
	defer p.valueMu.Unlock()
	p.Matrix = value
}

// ApplyDelta updates the value applying the delta.
func (p *BaseParam) ApplyDelta(delta mat.Matrix) {
	p.valueMu.Lock()
	defer p.valueMu.Unlock()
	p.Matrix.SubInPlace(delta)
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
