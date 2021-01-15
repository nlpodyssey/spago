// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
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
	ScaleFactor   mat.Float
	UseCausalMask bool
}

// ContextProb is a pair of Context encodings and Prob attention scores.
type ContextProb struct {
	// Context encodings.
	Context []ag.Node
	// Prob attention scores.
	Prob []mat.Matrix
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	return &Model{
		Config: config,
		Query:  linear.New(config.InputSize, config.QuerySize),
		Key:    linear.New(config.InputSize, config.KeySize),
		Value:  linear.New(config.InputSize, config.ValueSize),
	}
}

// Forward performs the forward step for each input node and returns the result.
// It generates the queries, keys and values from the same input xs.
//
// Valid input types: []ag.Node or nn.AttentionInput.
func (m *Model) Forward(attIn attention.QKV) []ag.Node {
	projAtt := attention.QKV{
		Queries: m.Query.Forward(attIn.Queries...),
		Keys:    m.Key.Forward(attIn.Keys...),
		Values:  m.Value.Forward(attIn.Values...),
	}
	context, prob := attention.ScaledDotProductAttention(m.Graph(), projAtt, m.ScaleFactor, m.UseCausalMask)
	m.Attention = &ContextProb{
		Context: context,
		Prob:    prob,
	}
	return context
}
