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
	opt       []interface{}
	model     *Model
	g         *ag.Graph
	mode      nn.ProcessingMode
	query     nn.Processor
	key       nn.Processor
	value     nn.Processor
	Attention *ContextProb
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:     m,
		g:         g,
		mode:      nn.Training,
		opt:       opt,
		query:     m.Query.NewProc(g),
		key:       m.Key.NewProc(g),
		value:     m.Value.NewProc(g),
		Attention: nil,
	}
	p.init(opt)
	return p
}

func (p *Processor) Model() nn.Model                { return p.model }
func (p *Processor) Graph() *ag.Graph               { return p.g }
func (p *Processor) RequiresFullSeq() bool          { return true }
func (p *Processor) Mode() nn.ProcessingMode        { return p.mode }
func (p *Processor) SetMode(mode nn.ProcessingMode) { p.mode = mode }

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("selfattention: invalid init options")
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	qs := p.query.Forward(xs...)
	ks := p.key.Forward(xs...)
	vs := p.value.Forward(xs...)
	context, prob := nn.ScaledDotProductAttention(p.g, qs, ks, vs, p.model.ScaleFactor)
	p.Attention = &ContextProb{
		context: context,
		prob:    prob,
	}
	return context
}
