// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stackedembeddings provides convenient types to stack multiple word embedding representations by concatenating them.
// The concatenation is then followed by a linear layer. The latter has the double utility of being able to project
// the concatenated embeddings in a smaller dimension, and to further train the final words representation.
package stackedembeddings

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

// WordsEncoderProcessor extends an nn.Processor providing the Encode method to
// transform a string sequence into an encoded representation.
type WordsEncoderProcessor interface {
	nn.Model
	// Encode transforms a string sequence into an encoded representation.
	Encode([]string) []ag.Node
}

var (
	_ nn.Model = &Model{}
)

// Model implements a stacked embeddings model.
// TODO: optional use of the projection layer?
// TODO: include an optional layer normalization?
type Model struct {
	nn.BaseModel
	WordsEncoders   []WordsEncoderProcessor
	ProjectionLayer *linear.Model
}

// Encode transforms a string sequence into an encoded representation.
func (m *Model) Encode(words []string) []ag.Node {
	encodingsPerWord := make([][]ag.Node, len(words))
	for _, encoder := range m.WordsEncoders {
		for wordIndex, encoding := range encoder.Encode(words) {
			encodingsPerWord[wordIndex] = append(encodingsPerWord[wordIndex], encoding)
		}
	}
	intermediateEncoding := make([]ag.Node, len(words))
	for wordIndex, encoding := range encodingsPerWord {
		if len(encoding) == 1 { // optimization
			intermediateEncoding[wordIndex] = encoding[0]
		} else {
			intermediateEncoding[wordIndex] = m.Graph().Concat(encoding...)
		}
	}
	return m.ProjectionLayer.Forward(intermediateEncoding...)
}
