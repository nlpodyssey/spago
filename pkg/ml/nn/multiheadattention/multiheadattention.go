// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multiheadattention

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/selfattention"
	"math"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Multi-Head Attention
type Model struct {
	Attention   []*selfattention.Model
	OutputMerge *linear.Model
	h           int // number of heads
	dm          int // input and output vectors dimension
	dk          int // hidden vectors dimension (dm/h)
}

func New(size, numOfHeads int) *Model {
	dm := size
	dk := size / numOfHeads
	attention := make([]*selfattention.Model, numOfHeads)
	attentionConfig := selfattention.Config{
		InputSize:   dm,
		QuerySize:   dk,
		KeySize:     dk,
		ValueSize:   dk,
		ScaleFactor: 1.0 / math.Sqrt(float64(dk)),
	}
	for i := 0; i < numOfHeads; i++ {
		attention[i] = selfattention.New(attentionConfig)
	}
	return &Model{
		Attention:   attention,
		OutputMerge: linear.New(dk*numOfHeads, dm),
		h:           numOfHeads,
		dm:          dm,
		dk:          dk,
	}
}

type Processor struct {
	nn.BaseProcessor
	HeadAttentionProc []*selfattention.Processor
	outputMerge       *linear.Processor
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	headAttentionProc := make([]*selfattention.Processor, m.h)
	for i := 0; i < m.h; i++ {
		headAttentionProc[i] = m.Attention[i].NewProc(g).(*selfattention.Processor)
	}
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		HeadAttentionProc: headAttentionProc,
		outputMerge:       m.OutputMerge.NewProc(g).(*linear.Processor),
	}
}

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	p.outputMerge.SetMode(mode)
	for _, proc := range p.HeadAttentionProc {
		proc.SetMode(mode)
	}
}

// Forward performs the the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	h := p.Model.(*Model).h
	headsAttention := make([][]ag.Node, h)
	for h, proc := range p.HeadAttentionProc {
		headsAttention[h] = proc.Forward(xs...)
	}
	concatHeads := make([]ag.Node, len(xs))
	for i := 0; i < len(xs); i++ {
		buf := make([]ag.Node, h)
		for j := 0; j < h; j++ {
			buf[j] = headsAttention[j][i]
		}
		concatHeads[i] = p.Graph.Concat(buf...)
	}
	return p.outputMerge.Forward(concatHeads...)
}
