// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stack

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the sub-models.
type Model struct {
	nn.BaseModel
	Layers []nn.StandardModel
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model.
func New(layers ...nn.StandardModel) *Model {
	return &Model{
		Layers: layers,
	}
}

// Make makes a new model by obtaining each layer with a callback.
func Make(size int, callback func(i int) nn.StandardModel) *Model {
	layers := make([]nn.StandardModel, size)
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
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := m.Layers[0].Forward(xs...)
	for i := 1; i < len(m.Layers); i++ {
		ys = m.Layers[i].Forward(ys...)
	}
	return ys
}
