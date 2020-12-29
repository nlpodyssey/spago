// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stack

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Layers []nn.Model
}

// New returns a new model.
func New(layers ...nn.Model) *Model {
	requireFullSeq := false
	for _, layer := range layers {
		if layer.RequiresCompleteSequence() {
			requireFullSeq = true
			break
		}
	}
	return &Model{
		BaseModel: nn.BaseModel{
			RCS: requireFullSeq,
		},
		Layers: layers,
	}
}

// Make makes a new model by obtaining each layer with a callback.
func Make(size int, callback func(i int) nn.Model) *Model {
	layers := make([]nn.Model, size)
	for i := 0; i < size; i++ {
		layers[i] = callback(i)
	}
	return New(layers...)
}

// LastLayer returns the last layer from the stack.
func (m *Model) LastLayer() nn.Model {
	return m.Layers[len(m.Layers)-1]
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(in interface{}) interface{} {
	xs := nn.ToNodes(in)
	if m.RequiresCompleteSequence() {
		return m.fullSeqForward(xs)
	}
	return m.incrementalForward(xs)
}

func (m *Model) fullSeqForward(xs []ag.Node) []ag.Node {
	ys := m.Layers[0].Forward(xs)
	for i := 1; i < len(m.Layers); i++ {
		ys = m.Layers[i].Forward(ys)
	}
	return ys.([]ag.Node)
}

func (m *Model) incrementalForward(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = m.singleForward(x)
	}
	return ys
}

func (m *Model) singleForward(x ag.Node) ag.Node {
	y := x
	for _, layer := range m.Layers {
		y = nn.ToNode(layer.Forward(y))
	}
	return y
}
