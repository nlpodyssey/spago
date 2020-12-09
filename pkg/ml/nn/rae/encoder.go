// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rae

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/pe"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Encoder{}
	_ nn.Processor = &EncoderProcessor{}
)

// Encoder contains the serializable parameters.
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

// NewProc returns a new processor to execute the forward step.
func (m *Encoder) NewProc(ctx nn.Context) nn.Processor {
	return &EncoderProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		ffn1:       m.ScalingFFN.NewProc(ctx),
		ffn2:       m.EncodingFFN.NewProc(ctx),
		recursions: 0,
	}
}

func (p *EncoderProcessor) GetRecursions() int {
	return p.recursions
}

// Forward performs the forward step for each input and returns the result.
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
	stepEncoder := p.Model.(*Encoder).StepEncoder
	stepEncoding := g.NewVariable(stepEncoder.EncodingAt(p.recursions), false)
	size := len(xs)
	ys := make([]ag.Node, size-1)
	for i := 0; i < size-1; i++ {
		ys[i] = g.Add(g.Concat(xs[i:i+2]...), stepEncoding)
	}
	return p.ffn2.Forward(ys...)
}
