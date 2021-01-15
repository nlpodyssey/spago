// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package birnncrf provides an implementation of a Bidirectional Recurrent Neural Network (BiRNN)
// with a Conditional Random Fields (CRF) on tom.
package birnncrf

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/crf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	BiRNN  *birnn.Model
	Scorer *linear.Model
	CRF    *crf.Model
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New(biRNN *birnn.Model, scorer *linear.Model, crf *crf.Model) *Model {
	return &Model{
		BiRNN:  biRNN,
		Scorer: scorer,
		CRF:    crf,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	return m.Scorer.Forward(m.BiRNN.Forward(xs...)...)
}

// Decode performs the viterbi decoding.
func (m *Model) Decode(emissionScores []ag.Node) []int {
	return m.CRF.Decode(emissionScores)
}

// Predict performs Decode(Forward(xs)).
func (m *Model) Predict(xs []ag.Node) []int {
	return m.Decode(m.Forward(xs...))
}

// NegativeLogLoss computes the negative log loss with respect to the targets.
// TODO: the CRF backward tests are still missing
func (m *Model) NegativeLogLoss(emissionScores []ag.Node, targets []int) ag.Node {
	return m.CRF.NegativeLogLoss(emissionScores, targets)
}
