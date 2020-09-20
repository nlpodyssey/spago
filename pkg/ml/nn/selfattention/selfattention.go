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
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	Config
	Query *linear.Model
	Key   *linear.Model
	Value *linear.Model
}

type Config struct {
	InputSize   int
	QuerySize   int
	KeySize     int
	ValueSize   int
	ScaleFactor float64
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

type ContextProb struct {
	context []ag.Node
	prob    []mat.Matrix
}

type Processor struct {
	nn.BaseProcessor
	scaleFactor float64
	query       *linear.Processor
	key         *linear.Processor
	value       *linear.Processor
	Attention   *ContextProb
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		scaleFactor: m.ScaleFactor,
		query:       m.Query.NewProc(g).(*linear.Processor),
		key:         m.Key.NewProc(g).(*linear.Processor),
		value:       m.Value.NewProc(g).(*linear.Processor),
		Attention:   nil,
	}
}

// Forward performs the the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	qs := p.query.Forward(xs...)
	ks := p.key.Forward(xs...)
	vs := p.value.Forward(xs...)
	context, prob := nn.ScaledDotProductAttention(p.Graph, qs, ks, vs, p.scaleFactor)
	p.Attention = &ContextProb{
		context: context,
		prob:    prob,
	}
	return context
}
