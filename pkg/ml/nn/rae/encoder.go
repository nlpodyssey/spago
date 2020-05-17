// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rae

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/pe"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
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

type EncoderProcessor struct {
	nn.BaseProcessor
	ffn1       nn.Processor
	ffn2       nn.Processor
	recursions int
}

func (m *Encoder) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &EncoderProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
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

func (p *EncoderProcessor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	p.ffn1.SetMode(mode)
	p.ffn2.SetMode(mode)
}

func (p *EncoderProcessor) GetRecursions() int {
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
	g := p.Graph
	stepEncoder := p.Model.(*Model).Decoder.StepEncoder
	stepEncoding := g.NewVariable(stepEncoder.EncodingAt(p.recursions), false)
	size := len(xs)
	ys := make([]ag.Node, size-1)
	for i := 0; i < size-1; i++ {
		ys[i] = g.Add(g.Concat(xs[i:i+2]...), stepEncoding)
	}
	return p.ffn2.Forward(ys...)
}
