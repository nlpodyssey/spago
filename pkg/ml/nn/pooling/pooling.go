// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pooling

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &MaxPooling{}
	_ nn.Processor = &Processor{}
)

// MaxPooling is a parameter-free model used to instantiate a new Processor.
type MaxPooling struct {
	Rows    int
	Columns int
}

// NewMax returns a new model.
func NewMax(rows, columns int) *MaxPooling {
	return &MaxPooling{}
}

type Processor struct {
	nn.BaseProcessor
}

// NewProc returns a new processor to execute the forward step.
func (m *MaxPooling) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
	}
}

// Forward performs the forward step for each input and returns the result.
// The max pooling is applied independently to each input.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	r, c := p.poolingDims()
	for i, x := range xs {
		ys[i] = p.Graph.MaxPooling(x, r, c)
	}
	return ys
}

func (p *Processor) poolingDims() (r, c int) {
	m := p.BaseProcessor.Model.(*MaxPooling)
	return m.Rows, m.Columns
}
