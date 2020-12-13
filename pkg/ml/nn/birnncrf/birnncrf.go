// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package birnncrf provides an implementation of a Bidirectional Recurrent Neural Network (BiRNN)
// with a Conditional Random Fields (CRF) on top.
package birnncrf

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/crf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

// Model contains the serializable parameters.
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

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		biRNN:      m.BiRNN.NewProc(ctx).(*birnn.Processor),
		scorer:     m.Scorer.NewProc(ctx).(*linear.Processor),
		lastScores: nil, // lazy initialized
	}
}

// Forward performs the forward step for each input and returns the result.
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
	decoder := p.Model.(*Model).CRF.NewProc(
		nn.Context{Graph: p.Graph, Mode: p.Mode}).(*crf.Processor)
	return decoder.NegativeLogLoss(p.lastScores, targets)
}
