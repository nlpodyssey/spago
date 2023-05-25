// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/gob"
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

type Param struct {
	mat.Matrix
	valueMu sync.RWMutex
	state   interface{}
	stateMu sync.RWMutex
}

// NewParam returns a new param.
func NewParam(value mat.Matrix) *Param {
	if value == nil {
		panic("nn: cannot create a new param with a nil value")
	}
	p := &Param{
		Matrix: value.Clone(),
		state:  nil,
	}
	p.SetRequiresGrad(true)
	return p
}

// WithGrad sets whether the param requires gradients.
// It is used to specify whether a Param should be trained or not.
func (p *Param) WithGrad(value bool) *Param {
	p.SetRequiresGrad(value)
	return p
}

// Value returns the value of the delegate itself.
func (p *Param) Value() mat.Matrix {
	p.valueMu.RLock()
	defer p.valueMu.RUnlock()
	return p.Matrix
}

// ReplaceValue replaces the value of the parameter.
// It also clears the gradients and the state.
func (p *Param) ReplaceValue(value mat.Matrix) {
	p.ClearState()
	p.ZeroGrad()

	p.valueMu.Lock()
	defer p.valueMu.Unlock()
	p.Matrix = value
}

// ApplyDelta updates the value applying the delta.
func (p *Param) ApplyDelta(delta mat.Matrix) {
	p.valueMu.Lock()
	defer p.valueMu.Unlock()
	p.Matrix.SubInPlace(delta)
}

// GetOrSetState returns the support structure for the optimizer.
func (p *Param) GetOrSetState(newStateFunc func(shape ...int) any) any {
	p.stateMu.RLock()
	defer p.stateMu.RUnlock()
	if p.state == nil && newStateFunc != nil {
		p.state = newStateFunc(p.Shape()...)
	}
	return p.state
}

// SetState sets the support structure for the optimizer.
func (p *Param) SetState(payload any) {
	p.stateMu.Lock()
	defer p.stateMu.Unlock()
	p.state = payload
}

// ClearState clears the support structure for the optimizer.
func (p *Param) ClearState() {
	p.stateMu.Lock()
	defer p.stateMu.Unlock()
	p.state = nil
}

func (p *Param) GobEncode() ([]byte, error) {
	p.valueMu.RLock()
	p.stateMu.RLock()
	defer p.valueMu.RUnlock()
	defer p.stateMu.RUnlock()

	gp := struct {
		Matrix mat.Matrix
		State  interface{}
	}{
		Matrix: p.Matrix,
		State:  p.state,
	}

	buf := new(bytes.Buffer)
	encoder := gob.NewEncoder(buf)
	if err := encoder.Encode(gp); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (p *Param) GobDecode(data []byte) error {
	gp := new(struct {
		Matrix mat.Matrix
		State  interface{}
	})

	buf := bytes.NewBuffer(data)
	decoder := gob.NewDecoder(buf)
	if err := decoder.Decode(gp); err != nil {
		return err
	}

	p.valueMu.Lock()
	p.stateMu.Lock()
	defer p.valueMu.Unlock()
	defer p.stateMu.Unlock()

	p.Matrix = gp.Matrix
	p.state = gp.State
	return nil
}
