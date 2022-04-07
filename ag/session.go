// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
)

// ProcessingMode regulates the different usage of some operations (e.g. Dropout, BatchNorm, etc.),
// depending on whether you're doing training or inference.
// Failing to set the right mode will yield inconsistent inference results.
type ProcessingMode uint8

const (
	// Training is to be used during the training phase of a model. For example, dropouts are enabled.
	Training ProcessingMode = iota
	// Inference keeps weights fixed while using the model and disables some operations (e.g. skip dropout).
	Inference
)

// SessionProvider provides the basic methods of a Session.
type SessionProvider[T mat.DType] interface {
	// Graph returns the graph used in this session
	Graph() *Graph[T]
	// Mode returns whether the graph is being used in training or inference.
	Mode() ProcessingMode
}

// Session encapsulates the Graph in which operations of a Differentiable module are executed.
// The Module() method returns a "reified" version of the same type of the input differentiable module,
// in which the fields of type Node (including those from other differentiable
// submodules) are connected to the session's graph.
type Session[T mat.DType, D Differentiable[T]] struct {
	module D
	graph  *Graph[T]
	mode   ProcessingMode
}

// NewSession construct a new session for the differentiable module working on a new Graph.
func NewSession[T mat.DType, D Differentiable[T]](i D, mode ProcessingMode) *Session[T, D] {
	g := NewGraph[T]()
	s := &Session[T, D]{
		graph: g,
		mode:  mode,
	}
	s.module = (&graphBinder[T]{session: s}).newBoundStruct(i).(Differentiable[T]).(D)
	return s
}

// Module returns a new differentiable structure of the same type as the one in input
// in which the fields of type Node (including those from other differentiable
// submodules) are connected to a graph.
func (s *Session[_, D]) Module() D {
	return s.module
}

// Graph returns the graph used in a session instance.
func (s *Session[T, _]) Graph() *Graph[T] {
	return s.graph
}

// Mode returns the mode used in a session instance.
func (s *Session[_, _]) Mode() ProcessingMode {
	return s.mode
}

// Close release resources associated with the Session.
// Trying to use the result of Module() after closing leads to panic,
// as there is no graph on which to perform operations.
func (s *Session[_, _]) Close() {
	s.graph.Clear()
	s.graph = nil
}

// NewVariable creates and returns a new variable owned by the session's graph.
func (s *Session[T, _]) NewVariable(value mat.Matrix[T], requiresGrad bool) Node[T] {
	return s.graph.NewVariable(value, requiresGrad)
}

// MarshalBinary satisfies encoding.BinaryMarshaler interface and prevents
// a Graph to be encoded to binary representation.
// This is relevant in the context of a Graph being part of a nn.Model: when
// serializing a model to binary, we want to skip the Graph, since it is part
// of the runtime context only.
func (s *Session[_, _]) MarshalBinary() ([]byte, error) {
	return []byte{}, nil
}

// UnmarshalBinary satisfies encoding.BinaryUnmarshaler interface.
func (s *Session[_, _]) UnmarshalBinary(_ []byte) error {
	return nil
}
