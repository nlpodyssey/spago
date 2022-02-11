// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package birnncrf provides an implementation of a Bidirectional Recurrent Neural Network (BiRNN)
// with a Conditional Random Fields (CRF) on tom.
package birnncrf

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/crf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	BiRNN  *birnn.Model[T]
	Scorer *linear.Model[T]
	CRF    *crf.Model[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](biRNN *birnn.Model[T], scorer *linear.Model[T], crf *crf.Model[T]) *Model[T] {
	return &Model[T]{
		BiRNN:  biRNN,
		Scorer: scorer,
		CRF:    crf,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	return m.Scorer.Forward(m.BiRNN.Forward(xs...)...)
}

// Decode performs the viterbi decoding.
func (m *Model[T]) Decode(emissionScores []ag.Node[T]) []int {
	return m.CRF.Decode(emissionScores)
}

// Predict performs Decode(Forward(xs)).
func (m *Model[T]) Predict(xs []ag.Node[T]) []int {
	return m.Decode(m.Forward(xs...))
}

// NegativeLogLoss computes the negative log loss with respect to the targets.
// TODO: the CRF backward tests are still missing
func (m *Model[T]) NegativeLogLoss(emissionScores []ag.Node[T], targets []int) ag.Node[T] {
	return m.CRF.NegativeLogLoss(emissionScores, targets)
}
