// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Reference: "Attention Is All You Need" by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin (2017)
(http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).
*/
package transformer

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var _ nn.Model = &Model{}

type Model struct {
	Config
	*stack.Model
}

type Config struct {
	size                   int
	numOfAttentionHeads    int
	intermediateSize       int
	intermediateActivation ag.OpName
	numOfLayers            int
	usePositionalEncoding  bool
	useDepthEncoding       bool
}

// New returns a new transformer model composed of a stack of N identical transformer layers.
func New(config Config) *Model {
	layers := make([]nn.Model, config.numOfLayers)
	for layerIndex := range layers {
		layers[layerIndex] = NewLayer(
			config.size,
			config.numOfAttentionHeads,
			config.intermediateSize,
			config.intermediateActivation,
			layerIndex,
			config.useDepthEncoding,
			config.usePositionalEncoding,
		)
	}
	return &Model{
		Config: config,
		Model:  stack.New(layers...),
	}
}

// LayerAt returns the layer model at the given index casted to the specific transformer layer model.
func LayerAt(transformer *Model, index int) *Layer {
	return transformer.Layers[index].(*Layer)
}

// LayerProcAt returns the layer processor at the given index casted to the specific transformer layer processor.
// It panics if the processor's model isn't a transformer.
func LayerProcAt(transformer *stack.Processor, index int) *LayerProcessor {
	if _, ok := transformer.Model().(*Model); ok {
		return transformer.Layers[index].(*LayerProcessor)
	} else {
		panic("transformer: invalid neural model")
	}
}
