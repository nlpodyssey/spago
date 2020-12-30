// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the activation operator and serializable parameters.
type Model struct {
	nn.BaseModel
	Activation ag.OpName
	Params     []nn.Param
}

// New returns a new model with parameters initialized to zeros.
// TODO: restrict operators to activation functions only; or create a dedicate builder for each activation.
func New(activation ag.OpName, params ...nn.Param) *Model {
	return &Model{
		Activation: activation,
		Params:     params,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	activation := m.Activation
	transformed := func(x ag.Node) ag.Node {
		return m.Graph().Invoke(activation, append([]ag.Node{x}, nn.Params(m.Params).Nodes()...)...)
	}
	return ag.Map(transformed, xs)
}
