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
	_ nn.Model = &Decoder{}
)

// Decoder contains the serializable parameters.
type Decoder struct {
	nn.BaseModel
	DecodingFNN1 nn.StandardModel // decoding part 1
	DecodingFFN2 nn.StandardModel // decoding part 2
	DescalingFFN nn.StandardModel
	StepEncoder  *pe.PositionalEncoder
	State        State `spago:"scope:processor"`
}

// State contains information used during decoding recursion.
type State struct {
	SequenceLength int
	MaxRecursions  int
	Recursions     int
}

// SetSequenceLength sets the length of the expected sequence.
func (p *Decoder) SetSequenceLength(length int) {
	maxRecursions := 0
	for i := 0; i < length-1; i++ {
		maxRecursions += i
	}
	p.State.SequenceLength = length
	p.State.MaxRecursions = maxRecursions
	p.State.Recursions = maxRecursions
}

// Forward performs the forward step for each input node and returns the result.
func (p *Decoder) Forward(x ag.Node) []ag.Node {
	// TODO: check the following commented code
	//if p.State.SequenceLength == 0 {
	//	log.Fatal("rae: sequence length must be > 0. Use p.SetSequenceLength().")
	//}
	//if len(xs) != p.State.SequenceLength {
	//	panic(fmt.Sprintf("rae: input sequence length is expected to be %d, but got %d", p.State.SequenceLength, len(xs)))
	//}
	ys := []ag.Node{x}
	for len(ys) != p.State.SequenceLength {
		ys = p.decodingStep(ys)
		p.State.Recursions--
	}
	return p.DescalingFFN.Forward(ys...)
}

func (p *Decoder) decodingStep(xs []ag.Node) []ag.Node {
	return p.DecodingFFN2.Forward(p.decodingPart1(xs)...)
}

func (p *Decoder) decodingPart1(xs []ag.Node) []ag.Node {
	decoding := p.splitVectors(p.DecodingFNN1.Forward(p.addStepEncoding(xs)...))
	ys := []ag.Node{decoding[0].y0, decoding[0].y1}
	for i := 1; i < len(xs); i++ {
		ys[i-1] = mean(p.Graph(), ys[i-1], decoding[i].y0)
		ys = append(ys, decoding[i].y1)
	}
	return ys
}

func (p *Decoder) addStepEncoding(xs []ag.Node) []ag.Node {
	g := p.Graph()
	stepEncoder := p.StepEncoder
	stepEncoding := g.NewVariable(stepEncoder.EncodingAt(p.State.Recursions), false)
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = g.Add(x, stepEncoding)
	}
	return ys
}

func (p *Decoder) splitVectors(xs []ag.Node) []struct{ y0, y1 ag.Node } {
	ys := make([]struct{ y0, y1 ag.Node }, len(xs))
	for i, x := range xs {
		lst := nn.SplitVec(p.Graph(), x, 2)
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
