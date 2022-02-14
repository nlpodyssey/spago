// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"sync"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// Param is the interface for a Model parameter.
type Param[T mat.DType] interface {
	ag.Node[T] // it implies fn.Operand and ag.GradValue too

	// Name returns the params name (can be empty string).
	Name() string
	// SetName set the params name (can be empty string).
	SetName(name string)
	// Type returns the params type (weights, biases, undefined).
	Type() ParamsType
	// SetType set the params type (weights, biases, undefined).
	SetType(pType ParamsType)
	// SetRequiresGrad set whether the param requires gradient, or not.
	SetRequiresGrad(value bool)
	// ReplaceValue replaces the value of the parameter and clears the support structure.
	ReplaceValue(value mat.Matrix[T])
	// ApplyDelta updates the value applying the delta.
	ApplyDelta(delta mat.Matrix[T])
	// Payload returns the optimizer support structure (can be nil).
	Payload() *Payload[T]
	// SetPayload is a thread safe operation to set the given Payload on the
	// receiver Param.
	SetPayload(payload *Payload[T])
	// ClearPayload clears the support structure.
	ClearPayload()
}

var _ Param[float32] = &param[float32]{}

type param[T mat.DType] struct {
	name         string
	pType        ParamsType    // lazy initialization
	mu           sync.Mutex    // to avoid data race
	value        mat.Matrix[T] // store the results of a forward evaluation.
	grad         mat.Matrix[T]
	payload      *Payload[T] // additional data used for example by gradient-descend optimization methods
	hasGrad      bool
	requiresGrad bool
}

// ParamOption allows to configure a new Param with your specific needs.
type ParamOption[T mat.DType] func(*param[T])

// RequiresGrad is an option to specify whether a Param should be trained or not.
func RequiresGrad[T mat.DType](value bool) ParamOption[T] {
	return func(p *param[T]) {
		p.requiresGrad = value
	}
}

// NewParam returns a new param.
func NewParam[T mat.DType](value mat.Matrix[T], opts ...ParamOption[T]) Param[T] {
	p := &param[T]{
		name:         "",        // lazy initialization
		pType:        Undefined, // lazy initialization
		value:        value,
		grad:         nil, // lazy initialization
		hasGrad:      false,
		requiresGrad: true, // true by default, can be modified with the options
		payload:      nil,  // lazy initialization
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// SetName set the params name (can be empty string).
func (p *param[_]) SetName(name string) {
	p.name = name
}

// SetType set the params type (weights, biases, undefined).
func (p *param[_]) SetType(pType ParamsType) {
	p.pType = pType
}

// Name returns the params name (can be empty string).
func (p *param[_]) Name() string {
	return p.name
}

// Type returns the params type (weights, biases, undefined).
func (p *param[_]) Type() ParamsType {
	return p.pType
}

// Value returns the value of the delegate itself.
func (p *param[T]) Value() mat.Matrix[T] {
	return p.value
}

// ReplaceValue replaces the value of the parameter and clears the support structure.
func (p *param[T]) ReplaceValue(value mat.Matrix[T]) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.value = value
	p.payload = nil
}

// ScalarValue returns the the scalar value of the node.
// It panics if the value is not a scalar.
// Note that it is not possible to start the backward step from a scalar value.
func (p *param[T]) ScalarValue() T {
	return p.value.Scalar()
}

// Grad returns the gradients accumulated during the backward pass.
func (p *param[T]) Grad() mat.Matrix[T] {
	return p.grad
}

// PropagateGrad accumulate the gradients
func (p *param[T]) PropagateGrad(grad mat.Matrix[T]) {
	if !p.requiresGrad {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.grad == nil {
		p.grad = p.value.ZerosLike()
	}
	p.grad.AddInPlace(grad)
	p.hasGrad = true
}

// HasGrad returns true if there are accumulated gradients.
func (p *param[_]) HasGrad() bool {
	return p.hasGrad
}

// RequiresGrad returns true if the param requires gradients.
func (p *param[_]) RequiresGrad() bool {
	return p.requiresGrad
}

// SetRequiresGrad is an option to specify whether a Param should be trained or not.
func (p *param[_]) SetRequiresGrad(value bool) {
	p.requiresGrad = value
}

// ZeroGrad clears the gradients.
func (p *param[_]) ZeroGrad() {
	if p.grad == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	defer mat.ReleaseMatrix(p.grad) //  release memory
	p.grad = nil
	p.hasGrad = false
}

// ApplyDelta updates the value applying the delta.
func (p *param[T]) ApplyDelta(delta mat.Matrix[T]) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.Value().SubInPlace(delta)
}

// Payload returns the optimizer support structure (can be nil).
func (p *param[T]) Payload() *Payload[T] {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.payload
}

// SetPayload is a thread safe operation to set the given Payload on the
// receiver Param.
func (p *param[T]) SetPayload(payload *Payload[T]) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.payload = payload
}

// ClearPayload clears the support structure.
func (p *param[_]) ClearPayload() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.payload = nil
}

// Graph returns always nil since the "pure" parameter is not associated with any graph.
func (p *param[T]) Graph() *ag.Graph[T] {
	panic("nn: attempting to access Graph on a not reified param.")
}

// ID returns always -1 since the "pure" parameter is not associated with any graph.
func (p *param[_]) ID() int {
	panic("nn: attempting to access the ID of a not reified param.")
}

// TimeStep returns always 0 since the "pure" parameter is not associated with any graph.
func (p *param[_]) TimeStep() int {
	panic("nn: attempting to access the TimeStep of a not reified param.")
}
