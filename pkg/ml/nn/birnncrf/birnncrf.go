// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Bidirectional Recurrent Neural Network (BiRNN) with a Conditional Random Fields (CRF) on top.
package birnncrf

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/crf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

type Model struct {
	BiRNN  *birnn.Model
	Scorer *linear.Model
	CRF    *crf.Model
}

type Processor struct {
	nn.BaseProcessor
	biRNN      *birnn.Processor
	scorer     *linear.Processor
	lastScores []ag.Node
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		biRNN:      m.BiRNN.NewProc(g).(*birnn.Processor),
		scorer:     m.Scorer.NewProc(g).(*linear.Processor),
		lastScores: nil, // lazy initialized
	}
}

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	nn.SetProcessingMode(mode, p.biRNN, p.scorer)
}

func (p Processor) Forward(xs ...ag.Node) []ag.Node {
	features := p.biRNN.Forward(xs...)
	return p.scorer.Forward(features...)
}

func (p *Processor) Predict(xs []ag.Node) []int {
	p.lastScores = p.Forward(xs...)
	return p.Model.(*Model).CRF.Predict(p.lastScores)
}

// TODO: the CRF backward tests are still missing
func (p *Processor) NegativeLogLoss(targets []int) ag.Node {
	decoder := p.Model.(*Model).CRF.NewProc(p.Graph).(*crf.Processor)
	return decoder.NegativeLogLoss(p.lastScores, targets)
}
