// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the activation operator and serializable parameters.
type Model struct {
	nn.Module
	Activation Name
	Params     []nn.Param
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model.
func New(activation Name, params ...nn.Param) *Model {
	return &Model{
		Activation: activation,
		Params:     params,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	if m.Activation == Identity {
		return xs
	}
	transformed := func(x ag.Node) ag.Node {
		return Do(m.Activation, append([]ag.Node{x}, ag.ToNodes(m.Params)...)...)
	}
	return ag.Map(transformed, xs)
}
