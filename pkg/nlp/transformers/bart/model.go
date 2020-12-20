// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bart implements the transformer model introduced by Mike et al., 2019.
// "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
// https://arxiv.org/abs/1910.13461
package bart

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartdecoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartencoder"
	"strconv"
)

// Model implements a BART model.
type Model struct {
	Config     bartconfig.Config
	Embeddings *embeddings.Model
	Encoder    *bartencoder.Model
	Decoder    *bartdecoder.Model
}

// New returns a new BART Model.
func New(config bartconfig.Config, embeddingsStoragePath string) *Model {
	return &Model{
		Config: config,
		Embeddings: embeddings.New(embeddings.Config{
			Size:       config.DModel,
			DBPath:     embeddingsStoragePath,
			ReadOnly:   !config.Training,
			ForceNewDB: false, // TODO: from config?
		}),
		Encoder: bartencoder.New(config),
		Decoder: bartdecoder.New(config),
	}
}

// Close closes the BART model's embeddings DB.
func (m *Model) Close() {
	m.Embeddings.Close()
}

// Processor implements the nn.Processor interface for a BART Model.
type Processor struct {
	nn.BaseProcessor
	bartconfig.Config
	Embeddings *embeddings.Processor
	Encoder    *bartencoder.Processor
	Decoder    *bartdecoder.Processor
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
		Embeddings: m.Embeddings.NewProc(ctx).(*embeddings.Processor),
		Encoder:    m.Encoder.NewProc(ctx).(*bartencoder.Processor),
		Decoder:    m.Decoder.NewProc(ctx).(*bartdecoder.Processor),
	}
}

// Process performs the forward step for each input and returns the result.
func (p *Processor) Process(inputIDs ...int) []ag.Node {
	encoderInput := p.Embeddings.Encode(intToStringSlice(inputIDs))
	encoderOutput := p.Encoder.Forward(encoderInput...)
	decoderInput := p.Embeddings.Encode(intToStringSlice(shiftR(inputIDs, 1)))
	decoderOutput := p.Decoder.Decode(decoderInput, encoderOutput)
	return decoderOutput
}

func intToStringSlice(a []int) []string {
	out := make([]string, len(a))
	for i, num := range a {
		out[i] = strconv.Itoa(num)
	}
	return out
}

func shiftR(a []int, i int) []int {
	x, b := a[:(len(a)-i)], a[(len(a)-i):]
	return append(b, x...)
}

// Forward is not implemented for BART model Processor (it always panics).
// You should use Process instead.
func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("bart: Forward() not implemented; use Process() instead.")
}
