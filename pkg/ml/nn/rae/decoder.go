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

type DecoderProcessor struct {
	opt            []interface{}
	model          *Decoder
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

func (p *DecoderProcessor) Model() nn.Model       { return p.model }
func (p *DecoderProcessor) Graph() *ag.Graph      { return p.g }
func (p *DecoderProcessor) RequiresFullSeq() bool { return true }
func (p *DecoderProcessor) Reset()                {}

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
	stepEncoding := p.g.NewVariable(p.model.StepEncoder.EncodingAt(p.recursions), false)
	a, b := p.decode(p.g.Add(xs[0], stepEncoding))
	ys := []ag.Node{a, b}
	for i := 1; i < len(xs); i++ {
		a, b := p.decode(p.g.Add(xs[i], stepEncoding))
		ys[i-1] = p.mean(ys[i-1], a)
		ys = append(ys, b)
	}

	return p.ffn2.Forward(ys...)
}

func (p *DecoderProcessor) decode(x ag.Node) (y0, y1 ag.Node) {
	y := nn.SplitVec(p.g, p.ffn1.Forward(x)[0], 2)
	return y[0], y[1]
}

func (p *DecoderProcessor) mean(x1 ag.Node, x2 ag.Node) ag.Node {
	return p.g.ProdScalar(p.g.Add(x1, x2), p.g.NewScalar(0.5))
}
