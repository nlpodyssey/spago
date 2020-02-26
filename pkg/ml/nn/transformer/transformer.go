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
	"saientist.dev/spago/pkg/ml/act"
	"saientist.dev/spago/pkg/ml/nn"
	"saientist.dev/spago/pkg/ml/nn/stack"
)

type Model struct {
	*stack.Model
}

// New returns a new transformer model composed of a stack of N identical transformer layers.
func New(size, numLayers, numAttentionHeads int, intermediateSize int, intermediateActivation act.FuncName) *Model {
	layers := make([]nn.Model, numLayers)
	for i := 0; i < numLayers; i++ {
		layers[i] = NewLayer(size, numAttentionHeads, intermediateSize, intermediateActivation)
	}
	return &Model{stack.New(layers...)}
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
