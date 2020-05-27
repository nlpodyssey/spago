// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of a sequence labeling architecture composed by Embeddings -> BiRNN -> Scorer -> CRF.
package sequencelabeler

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnncrf"
	"github.com/nlpodyssey/spago/pkg/nlp/stackedembeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	EmbeddingsLayer *stackedembeddings.Model
	TaggerLayer     *birnncrf.Model
	Labels          []string
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		EmbeddingsLayer: m.EmbeddingsLayer.NewProc(g).(*stackedembeddings.Processor),
		TaggerLayer:     m.TaggerLayer.NewProc(g).(*birnncrf.Processor),
	}
}

type Processor struct {
	nn.BaseProcessor
	EmbeddingsLayer *stackedembeddings.Processor
	TaggerLayer     *birnncrf.Processor
}

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	nn.SetProcessingMode(mode, p.EmbeddingsLayer, p.TaggerLayer)
}

type TokenLabel struct {
	tokenizers.StringOffsetsPair
	Label string
}

func (p *Processor) Predict(tokens []tokenizers.StringOffsetsPair) []TokenLabel {
	model := p.Model.(*Model)
	words := tokenizers.GetStrings(tokens)
	encodings := p.EmbeddingsLayer.Encode(words)
	prediction := p.TaggerLayer.Predict(encodings)
	result := make([]TokenLabel, len(tokens))
	for i, labelIndex := range prediction {
		result[i] = TokenLabel{
			StringOffsetsPair: tokens[i],
			Label:             model.Labels[labelIndex],
		}
	}
	return result
}

// TODO: it could be more consistent if the targets were the string labels
func (p *Processor) NegativeLogLoss(targets []int) ag.Node {
	return p.TaggerLayer.NegativeLogLoss(targets)
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("sequencetagger: method not implemented. Use Predict() instead.")
}
