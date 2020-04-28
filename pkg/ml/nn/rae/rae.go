// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Implementation of the recursive auto-encoder strategy described in "Towards Lossless Encoding of Sentences" by Prato et al., 2019.
Unlike the method described in the paper above, here I opted to use the positional encoding introduced by Vaswani et al. (2017) for the step encoding.
*/
package rae

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"io"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	Encoder *Encoder
	Decoder *Decoder
}

func New(encoder *Encoder, decoder *Decoder) *Model {
	return &Model{
		Encoder: encoder,
		Decoder: decoder,
	}
}

func (m *Model) ForEachParam(callback func(param *nn.Param)) {
	nn.ForEachParam(m, callback)
}

func (m *Model) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

func (m *Model) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

type Processor struct {
	opt            []interface{}
	model          *Model
	mode           nn.ProcessingMode
	g              *ag.Graph
	Encoder        *EncoderProcessor
	Decoder        *DecoderProcessor
	SequenceLength int
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	if len(opt) != 1 {
		log.Fatal("rae: missing sequence length argument")
	}
	sequenceLength := 0
	if t, ok := opt[0].(int); ok {
		sequenceLength = t
	} else {
		log.Fatal("rae: invalid init options; sequence length was expected")
	}

	return &Processor{
		model:          m,
		mode:           nn.Training,
		opt:            opt,
		g:              g,
		Encoder:        m.Encoder.NewProc(g).(*EncoderProcessor),
		Decoder:        m.Decoder.NewProc(g, sequenceLength).(*DecoderProcessor),
		SequenceLength: sequenceLength,
	}
}

func (p *Processor) Model() nn.Model         { return p.model }
func (p *Processor) Graph() *ag.Graph        { return p.g }
func (p *Processor) RequiresFullSeq() bool   { return true }
func (p *Processor) Mode() nn.ProcessingMode { return p.mode }

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.mode = mode
	p.Encoder.SetMode(mode)
	p.Decoder.SetMode(mode)
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	if len(xs) != p.SequenceLength {
		panic(fmt.Sprintf("rae: input sequence length is expected to be %d, but got %d", p.SequenceLength, len(xs)))
	}
	enc := p.Encoder.Forward(xs...)
	dec := p.Decoder.Forward(enc...)
	return dec
}
