// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crf

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Size             int
	TransitionScores nn.Param[T]    `spago:"type:weights"`
	Scores           [][]ag.Node[T] `spago:"scope:processor"`
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new convolution Model, initialized according to the given configuration.
func New[T mat.DType](size int) *Model[T] {
	return &Model[T]{
		Size:             size,
		TransitionScores: nn.NewParam[T](mat.NewEmptyDense[T](size+1, size+1)), // +1 for start and end transitions
	}
}

// InitProcessor initializes structures and data useful for the decoding.
func (m *Model[T]) InitProcessor() {
	m.Scores = nn.SeparateMatrix[T](m.Graph(), m.TransitionScores) // TODO: lazy initialization
}

// Decode performs viterbi decoding.
func (m *Model[T]) Decode(emissionScores []ag.Node[T]) []int {
	return Viterbi(m.TransitionScores.Value(), emissionScores)
}

// NegativeLogLoss computes the negative log loss with respect to the targets.
func (m *Model[T]) NegativeLogLoss(emissionScores []ag.Node[T], target []int) ag.Node[T] {
	goldScore := m.goldScore(emissionScores, target)
	totalScore := m.totalScore(emissionScores)
	return m.Graph().Sub(totalScore, goldScore)
}

func (m *Model[T]) goldScore(emissionScores []ag.Node[T], target []int) ag.Node[T] {
	g := m.Graph()
	goldScore := g.At(emissionScores[0], target[0], 0)
	goldScore = g.Add(goldScore, m.Scores[0][target[0]+1]) // start transition
	prevIndex := target[0] + 1
	for i := 1; i < len(target); i++ {
		goldScore = g.Add(goldScore, g.AtVec(emissionScores[i], target[i]))
		goldScore = g.Add(goldScore, m.Scores[prevIndex][target[i]+1])
		prevIndex = target[i] + 1
	}
	goldScore = g.Add(goldScore, m.Scores[prevIndex][0]) // end transition
	return goldScore
}

func (m *Model[T]) totalScore(predicted []ag.Node[T]) ag.Node[T] {
	g := m.Graph()
	totalVector := m.totalScoreStart(predicted[0])
	for i := 1; i < len(predicted); i++ {
		totalVector = m.totalScoreStep(totalVector, nn.SeparateVec(g, predicted[i]))
	}
	totalVector = m.totalScoreEnd(totalVector)
	return g.Log(g.ReduceSum(g.Concat(totalVector...)))

}

func (m *Model[T]) totalScoreStart(stepVec ag.Node[T]) []ag.Node[T] {
	firstTransitionScores := m.Scores[0]
	scores := make([]ag.Node[T], m.Size)
	g := m.Graph()
	for i := 0; i < m.Size; i++ {
		scores[i] = g.Add(g.AtVec(stepVec, i), firstTransitionScores[i+1])
	}
	return scores
}

func (m *Model[T]) totalScoreEnd(stepVec []ag.Node[T]) []ag.Node[T] {
	scores := make([]ag.Node[T], m.Size)
	g := m.Graph()
	for i := 0; i < m.Size; i++ {
		vecTrans := g.Add(stepVec[i], m.Scores[i+1][0])
		scores[i] = g.Add(scores[i], g.Exp(vecTrans))
	}
	return scores
}

func (m *Model[T]) totalScoreStep(totalVec []ag.Node[T], stepVec []ag.Node[T]) []ag.Node[T] {
	scores := make([]ag.Node[T], m.Size)
	g := m.Graph()
	for i := 0; i < m.Size; i++ {
		nodei := totalVec[i]
		transitionScores := m.Scores[i+1]
		for j := 0; j < m.Size; j++ {
			vecSum := g.Add(nodei, stepVec[j])
			vecTrans := g.Add(vecSum, transitionScores[j+1])
			scores[j] = g.Add(scores[j], g.Exp(vecTrans))
		}
	}
	for i := 0; i < m.Size; i++ {
		scores[i] = g.Log(scores[i])
	}
	return scores
}
