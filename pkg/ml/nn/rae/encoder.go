// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rae

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/pe"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &Encoder{}
)

// Encoder contains the serializable parameters.
type Encoder struct {
	nn.BaseModel
	ScalingFFN  nn.StandardModel
	EncodingFFN nn.StandardModel
	StepEncoder *pe.PositionalEncoder
	Recursions  int `spago:"scope:processor"`
}

func init() {
	gob.Register(&Encoder{})
}

// GetRecursions returns the number of recursions.
func (p *Encoder) GetRecursions() int {
	return p.Recursions
}

// Forward performs the forward step for each input node and returns the result.
func (p *Encoder) Forward(xs ...ag.Node) []ag.Node {
	ys := p.ScalingFFN.Forward(xs...)
	p.Recursions = 1
	for len(ys) > 1 {
		ys = p.encodingStep(ys)
		p.Recursions++
	}
	return ys
}

func (p *Encoder) encodingStep(xs []ag.Node) []ag.Node {
	g := p.Graph()
	stepEncoder := p.StepEncoder
	stepEncoding := g.NewVariable(stepEncoder.EncodingAt(p.Recursions), false)
	size := len(xs)
	ys := make([]ag.Node, size-1)
	for i := 0; i < size-1; i++ {
		ys[i] = g.Add(g.Concat(xs[i:i+2]...), stepEncoding)
	}
	return p.EncodingFFN.Forward(ys...)
}
