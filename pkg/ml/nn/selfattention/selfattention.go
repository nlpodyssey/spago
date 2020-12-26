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
	_ nn.Module = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config
	Query     *linear.Model `type:"param"`
	Key       *linear.Model `type:"param"`
	Value     *linear.Model `type:"param"`
	Attention *ContextProb  `scope:"processor"`
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
		BaseModel: nn.BaseModel{FullSeqProcessing: true},
		Config:    config,
		Query:     linear.New(config.InputSize, config.QuerySize),
		Key:       linear.New(config.InputSize, config.KeySize),
		Value:     linear.New(config.InputSize, config.ValueSize),
	}
}

// Forward performs the forward step for each input and returns the result.
// It generates the queries, keys and values from the same input xs.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	qs := m.Query.Forward(xs...)
	ks := m.Key.Forward(xs...)
	vs := m.Value.Forward(xs...)
	context, prob := nn.ScaledDotProductAttention(m.GetGraph(), qs, ks, vs, m.ScaleFactor, m.UseCausalMask)
	m.Attention = &ContextProb{
		Context: context,
		Prob:    prob,
	}
	return context
}

// ForwardQKV performs the forward step for each input and returns the result.
func (m *Model) ForwardQKV(qs []ag.Node, ks []ag.Node, vs []ag.Node) []ag.Node {
	qsProj := m.Query.Forward(qs...)
	ksProj := m.Key.Forward(ks...)
	vsProj := m.Value.Forward(vs...)
	context, prob := nn.ScaledDotProductAttention(m.GetGraph(), qsProj, ksProj, vsProj, m.ScaleFactor, m.UseCausalMask)
	m.Attention = &ContextProb{
		Context: context,
		Prob:    prob,
	}
	return context
}
