// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stackedembeddings provides convenient types to stack multiple word embedding representations by concatenating them.
// The concatenation is then followed by a linear layer. The latter has the double utility of being able to project
// the concatenated embeddings in a smaller dimension, and to further train the final words representation.
package stackedembeddings

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"log"
)

type WordsEncoderProcessor interface {
	nn.Processor
	Encode([]string) []ag.Node
}

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// TODO: optional use of the projection layer?
// TODO: include an optional layer normalization?
type Model struct {
	WordsEncoders   []nn.Model
	ProjectionLayer *linear.Model
}

func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	processors := make([]WordsEncoderProcessor, len(m.WordsEncoders))
	for i, encoder := range m.WordsEncoders {
		proc, ok := encoder.NewProc(ctx).(WordsEncoderProcessor)
		if !ok {
			log.Fatal(fmt.Sprintf(
				"stackedembeddings: impossible to instantiate a `WordsEncoderProcessor` at index %d", i))
		}
		processors[i] = proc
	}
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		encoders:        processors,
		projectionLayer: m.ProjectionLayer.NewProc(ctx).(*linear.Processor),
	}
}

type Processor struct {
	nn.BaseProcessor
	encoders        []WordsEncoderProcessor
	projectionLayer *linear.Processor
}

func (p *Processor) Encode(words []string) []ag.Node {
	encodingsPerWord := make([][]ag.Node, len(words))
	for _, encoder := range p.encoders {
		for wordIndex, encoding := range encoder.Encode(words) {
			encodingsPerWord[wordIndex] = append(encodingsPerWord[wordIndex], encoding)
		}
	}
	intermediateEncoding := make([]ag.Node, len(words))
	for wordIndex, encoding := range encodingsPerWord {
		if len(encoding) == 1 { // optimization
			intermediateEncoding[wordIndex] = encoding[0]
		} else {
			intermediateEncoding[wordIndex] = p.Graph.Concat(encoding...)
		}
	}
	return p.projectionLayer.Forward(intermediateEncoding...)
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("stackedembeddings: method not implemented. Use Encode() instead.")
}
