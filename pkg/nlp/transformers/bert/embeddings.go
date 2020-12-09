// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
)

var (
	_ nn.Model     = &Embeddings{}
	_ nn.Processor = &EmbeddingsProcessor{}
)

type EmbeddingsConfig struct {
	Size                int
	OutputSize          int
	MaxPositions        int
	TokenTypes          int
	WordsMapFilename    string
	WordsMapReadOnly    bool
	DeletePreEmbeddings bool
}

type Embeddings struct {
	EmbeddingsConfig
	Word      *embeddings.Model
	Position  []*nn.Param `type:"weights"`
	TokenType []*nn.Param `type:"weights"`
	Norm      *layernorm.Model
	Projector *linear.Model
}

func NewEmbeddings(config EmbeddingsConfig) *Embeddings {
	return &Embeddings{
		EmbeddingsConfig: config,
		Word: embeddings.New(embeddings.Config{
			Size:       config.Size,
			DBPath:     config.WordsMapFilename,
			ReadOnly:   config.WordsMapReadOnly,
			ForceNewDB: config.DeletePreEmbeddings,
		}),
		Position:  newPositionEmbeddings(config.Size, config.MaxPositions),
		TokenType: newTokenTypes(config.Size, config.TokenTypes),
		Norm:      layernorm.New(config.Size),
		Projector: newProjector(config.Size, config.OutputSize),
	}
}

func newPositionEmbeddings(size, maxPositions int) []*nn.Param {
	out := make([]*nn.Param, maxPositions)
	for i := 0; i < maxPositions; i++ {
		out[i] = nn.NewParam(mat.NewEmptyVecDense(size))
	}
	return out
}

func newTokenTypes(size, tokenTypes int) []*nn.Param {
	out := make([]*nn.Param, tokenTypes)
	for i := 0; i < tokenTypes; i++ {
		out[i] = nn.NewParam(mat.NewEmptyVecDense(size))
	}
	return out
}

func newProjector(in, out int) *linear.Model {
	if in == out {
		return nil // projection not needed
	}
	return linear.New(in, out)
}

type EmbeddingsProcessor struct {
	nn.BaseProcessor
	model               *Embeddings
	wordsLayer          *embeddings.Processor
	norm                *layernorm.Processor
	projection          *linear.Processor
	tokenTypeEmbeddings []ag.Node
	unknownEmbedding    ag.Node
}

func (m *Embeddings) NewProc(ctx nn.Context) nn.Processor {
	var projection *linear.Processor = nil
	if m.Projector != nil {
		projection = m.Projector.NewProc(ctx).(*linear.Processor)
	}
	tokenTypeEmbeddings := make([]ag.Node, m.TokenTypes)
	for i, param := range m.TokenType {
		tokenTypeEmbeddings[i] = ctx.Graph.NewWrap(param)
	}
	return &EmbeddingsProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		model:               m,
		wordsLayer:          m.Word.NewProc(ctx).(*embeddings.Processor),
		norm:                m.Norm.NewProc(ctx).(*layernorm.Processor),
		projection:          projection,
		tokenTypeEmbeddings: tokenTypeEmbeddings,
		unknownEmbedding:    ctx.Graph.NewWrap(m.Word.GetEmbedding(wordpiecetokenizer.DefaultUnknownToken)),
	}
}

func (p *EmbeddingsProcessor) Encode(words []string) []ag.Node {
	encoded := make([]ag.Node, len(words))
	wordEmbeddings := p.getWordEmbeddings(words)
	sequenceIndex := 0
	for i := 0; i < len(words); i++ {
		encoded[i] = wordEmbeddings[i]
		encoded[i] = p.Graph.Add(encoded[i], p.Graph.NewWrap(p.model.Position[i]))
		encoded[i] = p.Graph.Add(encoded[i], p.tokenTypeEmbeddings[sequenceIndex])
		if words[i] == wordpiecetokenizer.DefaultSequenceSeparator {
			sequenceIndex++
		}
	}
	return p.useProjection(p.norm.Forward(encoded...))
}

func (p *EmbeddingsProcessor) getWordEmbeddings(words []string) []ag.Node {
	out := make([]ag.Node, len(words))
	for i, embedding := range p.wordsLayer.Encode(words) {
		switch embedding {
		case nil:
			out[i] = p.unknownEmbedding
		default:
			out[i] = embedding
		}
	}
	return out
}

func (p *EmbeddingsProcessor) useProjection(xs []ag.Node) []ag.Node {
	if p.projection == nil {
		return xs
	}
	return p.projection.Forward(xs...)
}

func (p *EmbeddingsProcessor) Forward(_ ...ag.Node) []ag.Node {
	panic("bert: Forward() method not implemented. Use Encode() instead.")
}
