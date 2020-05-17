// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Self-Attention
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

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
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
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("selfattention: invalid init options")
	}
}

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
