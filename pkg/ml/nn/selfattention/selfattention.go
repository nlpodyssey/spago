// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config
	Query     *linear.Model
	Key       *linear.Model
	Value     *linear.Model
	Attention *ContextProb `spago:"scope:processor"`
}

// Config provides configuration settings for a Self-Attention Model.
type Config struct {
	InputSize     int
	QuerySize     int
	KeySize       int
	ValueSize     int
	ScaleFactor   float64
	UseCausalMask bool
}

// ContextProb is a pair of Context encodings and Prob attention scores.
type ContextProb struct {
	// Context encodings.
	Context []ag.Node
	// Prob attention scores.
	Prob []mat.Matrix
}

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	return &Model{
		BaseModel: nn.BaseModel{RCS: true},
		Config:    config,
		Query:     linear.New(config.InputSize, config.QuerySize),
		Key:       linear.New(config.InputSize, config.KeySize),
		Value:     linear.New(config.InputSize, config.ValueSize),
	}
}

// Forward performs the forward step for each input node and returns the result.
// It generates the queries, keys and values from the same input xs.
//
// Valid input types: []ag.Node or nn.AttentionInput.
func (m *Model) Forward(in interface{}) interface{} {
	attIn, isAttentionInput := in.(nn.AttentionInput)
	if !isAttentionInput {
		nodes := nn.ToNodes(in)
		attIn = nn.AttentionInput{Queries: nodes, Keys: nodes, Values: nodes}
	}

	projAtt := nn.AttentionInput{
		Queries: m.Query.Forward(attIn.Queries).([]ag.Node),
		Keys:    m.Key.Forward(attIn.Keys).([]ag.Node),
		Values:  m.Value.Forward(attIn.Values).([]ag.Node),
	}

	context, prob := nn.ScaledDotProductAttention(m.Graph(), projAtt, m.ScaleFactor, m.UseCausalMask)
	m.Attention = &ContextProb{
		Context: context,
		Prob:    prob,
	}
	return context
}
