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

var _ nn.Model = &Decoder{}

type Decoder struct {
	DecodingFNN1 nn.Model // decoding part 1
	DecodingFFN2 nn.Model // decoding part 2
	DescalingFFN nn.Model
	StepEncoder  *pe.PositionalEncoder
}

func (m *Decoder) ForEachParam(callback func(param *nn.Param)) {
	nn.ForEachParam(m, callback)
}

func (m *Decoder) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

func (m *Decoder) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

var _ nn.Processor = &DecoderProcessor{}

type DecoderProcessor struct {
	opt            []interface{}
	model          *Decoder
	mode           nn.ProcessingMode
	g              *ag.Graph
	ffn1           nn.Processor
	ffn2           nn.Processor
	ffn3           nn.Processor
	sequenceLength int
	maxRecursions  int
	recursions     int
}

func (m *Decoder) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	if len(opt) != 1 {
		log.Fatal("rae: missing sequence length argument")
	}
	sequenceLength := 0
	if t, ok := opt[0].(int); ok {
		sequenceLength = t
	} else {
		log.Fatal("rae: invalid init options; sequence length was expected")
	}

	maxRecursions := 0
	for i := 0; i < sequenceLength-1; i++ {
		maxRecursions += i
	}

	return &DecoderProcessor{
		model:          m,
		mode:           nn.Training,
		opt:            opt,
		g:              g,
		ffn1:           m.DecodingFNN1.NewProc(g),
		ffn2:           m.DecodingFFN2.NewProc(g),
		ffn3:           m.DescalingFFN.NewProc(g),
		sequenceLength: sequenceLength,
		maxRecursions:  maxRecursions,
		recursions:     maxRecursions,
	}
}

func (p *DecoderProcessor) Model() nn.Model         { return p.model }
func (p *DecoderProcessor) Graph() *ag.Graph        { return p.g }
func (p *DecoderProcessor) RequiresFullSeq() bool   { return true }
func (p *DecoderProcessor) Mode() nn.ProcessingMode { return p.mode }
func (p *DecoderProcessor) Reset()                  {}

func (p *DecoderProcessor) SetMode(mode nn.ProcessingMode) {
	p.mode = mode
	p.ffn1.SetMode(mode)
	p.ffn2.SetMode(mode)
	p.ffn3.SetMode(mode)
}

func (p *DecoderProcessor) Forward(xs ...ag.Node) []ag.Node {
	if len(xs) != 1 {
		panic("rae: the input must be a single node")
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
		ys[i-1] = p.mean(ys[i-1], decoding[i].y0)
		ys = append(ys, decoding[i].y1)
	}
	return ys
}

func (p *DecoderProcessor) addStepEncoding(xs []ag.Node) []ag.Node {
	stepEncoding := p.g.NewVariable(p.model.StepEncoder.EncodingAt(p.recursions), false)
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.g.Add(x, stepEncoding)
	}
	return ys
}

func (p *DecoderProcessor) splitVectors(xs []ag.Node) []struct{ y0, y1 ag.Node } {
	ys := make([]struct{ y0, y1 ag.Node }, len(xs))
	for i, x := range xs {
		lst := nn.SplitVec(p.g, x, 2)
		ys[i] = struct{ y0, y1 ag.Node }{
			y0: lst[0],
			y1: lst[1],
		}
	}
	return ys
}

func (p *DecoderProcessor) mean(x1 ag.Node, x2 ag.Node) ag.Node {
	return p.g.ProdScalar(p.g.Add(x1, x2), p.g.NewScalar(0.5))
}
