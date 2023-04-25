// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mlpmixer

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
)

// FeedForward is the model for feed-forward operations of a MixerBlock.
type FeedForward struct {
	nn.Module
	Layers nn.ModuleList[nn.StandardModel]
}

func newFeedForward[T float.DType](dim, hiddenDim int, act activation.Name, dropout T) *FeedForward {
	return &FeedForward{
		Layers: []nn.StandardModel{
			linear.New[T](dim, hiddenDim),
			activation.New(act),
			// dropout.New(dropout),
			linear.New[T](hiddenDim, dim),
			// dropout.New(dropout),
		},
	}
}

func (m *FeedForward) Forward(xs ...ag.Node) []ag.Node {
	return m.Layers.Forward(xs...)
}
