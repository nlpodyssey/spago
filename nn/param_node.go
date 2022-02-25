// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

var _ Param[float32] = &ParamNode[float32]{}

// ParamNode enriches a Param with a Node.
type ParamNode[T mat.DType] struct {
	Param[T]
	Node ag.Node[T]
}

// ID dispatches the call to the Node.
func (p *ParamNode[_]) ID() int {
	return p.Node.ID()
}

// TimeStep dispatches the call to the Node.
func (p *ParamNode[_]) TimeStep() int {
	return p.Node.TimeStep()
}

// Graph dispatches the call to the Node.
func (p *ParamNode[T]) Graph() *ag.Graph[T] {
	return p.Node.Graph()
}

// Grad dispatches the call to the Node.
func (p *ParamNode[T]) Grad() mat.Matrix[T] {
	return p.Node.Grad()
}

// PropagateGrad dispatches the call to the Node.
func (p *ParamNode[T]) PropagateGrad(gx mat.Matrix[T]) {
	p.Node.PropagateGrad(gx)
}

// HasGrad dispatches the call to the Node.
func (p *ParamNode[_]) HasGrad() bool {
	return p.Node.HasGrad()
}

// RequiresGrad dispatches the call to the Node.
func (p *ParamNode[_]) RequiresGrad() bool {
	return p.Node.RequiresGrad()
}

// ZeroGrad dispatches the call to the Node.
func (p *ParamNode[_]) ZeroGrad() {
	p.Node.ZeroGrad()
}
