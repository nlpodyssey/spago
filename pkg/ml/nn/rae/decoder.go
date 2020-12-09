// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rae

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/pe"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model     = &Decoder{}
	_ nn.Processor = &DecoderProcessor{}
)

// Decoder contains the serializable parameters.
type Decoder struct {
	DecodingFNN1 nn.Model // decoding part 1
	DecodingFFN2 nn.Model // decoding part 2
	DescalingFFN nn.Model
	StepEncoder  *pe.PositionalEncoder
}

type DecoderProcessor struct {
	nn.BaseProcessor
	ffn1           nn.Processor
	ffn2           nn.Processor
	ffn3           nn.Processor
	sequenceLength int
	maxRecursions  int
	recursions     int
}

func (p *DecoderProcessor) SetSequenceLength(length int) {
	maxRecursions := 0
	for i := 0; i < length-1; i++ {
		maxRecursions += i
	}
	p.sequenceLength = length
	p.maxRecursions = maxRecursions
	p.recursions = maxRecursions
}

// NewProc returns a new processor to execute the forward step.
func (m *Decoder) NewProc(ctx nn.Context) nn.Processor {
	return &DecoderProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		ffn1:           m.DecodingFNN1.NewProc(ctx),
		ffn2:           m.DecodingFFN2.NewProc(ctx),
		ffn3:           m.DescalingFFN.NewProc(ctx),
		sequenceLength: 0, // late init
		maxRecursions:  0, // late init
		recursions:     0, // late init
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *DecoderProcessor) Forward(xs ...ag.Node) []ag.Node {
	if len(xs) != 1 {
		panic("rae: the input must be a single node")
	}
	if p.sequenceLength == 0 {
		log.Fatal("rae: sequence length must be > 0. Use p.SetSequenceLength().")
	}
	if len(xs) != p.sequenceLength {
		panic(fmt.Sprintf("rae: input sequence length is expected to be %d, but got %d", p.sequenceLength, len(xs)))
	}
	ys := []ag.Node{xs[0]}
	for len(ys) != p.sequenceLength {
		ys = p.decodingStep(ys)
		p.recursions--
	}
	return p.ffn3.Forward(ys...)
}

func (p *DecoderProcessor) decodingStep(xs []ag.Node) []ag.Node {
	return p.ffn2.Forward(p.decodingPart1(xs)...)
}

func (p *DecoderProcessor) decodingPart1(xs []ag.Node) []ag.Node {
	decoding := p.splitVectors(p.ffn1.Forward(p.addStepEncoding(xs)...))
	ys := []ag.Node{decoding[0].y0, decoding[0].y1}
	for i := 1; i < len(xs); i++ {
		ys[i-1] = mean(p.Graph, ys[i-1], decoding[i].y0)
		ys = append(ys, decoding[i].y1)
	}
	return ys
}

func (p *DecoderProcessor) addStepEncoding(xs []ag.Node) []ag.Node {
	g := p.Graph
	stepEncoder := p.Model.(*Decoder).StepEncoder
	stepEncoding := g.NewVariable(stepEncoder.EncodingAt(p.recursions), false)
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = g.Add(x, stepEncoding)
	}
	return ys
}

func (p *DecoderProcessor) splitVectors(xs []ag.Node) []struct{ y0, y1 ag.Node } {
	ys := make([]struct{ y0, y1 ag.Node }, len(xs))
	for i, x := range xs {
		lst := nn.SplitVec(p.Graph, x, 2)
		ys[i] = struct{ y0, y1 ag.Node }{
			y0: lst[0],
			y1: lst[1],
		}
	}
	return ys
}

func mean(g *ag.Graph, x1 ag.Node, x2 ag.Node) ag.Node {
	return g.ProdScalar(g.Add(x1, x2), g.NewScalar(0.5))
}
