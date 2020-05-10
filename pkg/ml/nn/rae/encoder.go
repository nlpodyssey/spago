// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rae

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/pe"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"io"
	"log"
)

var (
	_ nn.Model     = &Encoder{}
	_ nn.Processor = &EncoderProcessor{}
)

type Encoder struct {
	ScalingFFN  nn.Model
	EncodingFFN nn.Model
	StepEncoder *pe.PositionalEncoder
}

func (m *Encoder) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

func (m *Encoder) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

type EncoderProcessor struct {
	opt        []interface{}
	model      *Encoder
	mode       nn.ProcessingMode
	g          *ag.Graph
	ffn1       nn.Processor
	ffn2       nn.Processor
	recursions int
}

func (m *Encoder) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &EncoderProcessor{
		model:      m,
		mode:       nn.Training,
		opt:        opt,
		g:          g,
		ffn1:       m.ScalingFFN.NewProc(g),
		ffn2:       m.EncodingFFN.NewProc(g),
		recursions: 0,
	}
	p.init(opt)
	return p
}

func (p *EncoderProcessor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("rae: invalid init options")
	}
}

func (p *EncoderProcessor) Model() nn.Model         { return p.model }
func (p *EncoderProcessor) Graph() *ag.Graph        { return p.g }
func (p *EncoderProcessor) RequiresFullSeq() bool   { return true }
func (p *EncoderProcessor) Mode() nn.ProcessingMode { return p.mode }

func (p *EncoderProcessor) SetMode(mode nn.ProcessingMode) {
	p.mode = mode
	p.ffn1.SetMode(mode)
	p.ffn2.SetMode(mode)
}

func (p *EncoderProcessor) Recursions() int {
	return p.recursions
}

func (p *EncoderProcessor) Forward(xs ...ag.Node) []ag.Node {
	ys := p.ffn1.Forward(xs...)
	p.recursions = 1
	for len(ys) > 1 {
		ys = p.encodingStep(ys)
		p.recursions++
	}
	return ys
}

func (p *EncoderProcessor) encodingStep(xs []ag.Node) []ag.Node {
	stepEncoding := p.g.NewVariable(p.model.StepEncoder.EncodingAt(p.recursions), false)
	size := len(xs)
	ys := make([]ag.Node, size-1)
	for i := 0; i < size-1; i++ {
		ys[i] = p.g.Add(p.g.Concat(xs[i:i+2]...), stepEncoding)
	}
	return p.ffn2.Forward(ys...)
}
