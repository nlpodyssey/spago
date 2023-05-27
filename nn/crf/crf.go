// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crf

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Size             int
	TransitionScores *nn.Param
}

func init() {
	gob.Register(&Model{})
}

// New returns a new convolution Model, initialized according to the given configuration.
func New[T float.DType](size int) *Model {
	return &Model{
		Size:             size,
		TransitionScores: nn.NewParam(mat.NewDense[T](mat.WithShape(size+1, size+1))), // +1 for start and end transitions
	}
}

// Decode performs viterbi decoding.
func (m *Model) Decode(emissionScores []ag.DualValue) []int {
	return Viterbi(m.TransitionScores.Value(), emissionScores)
}

// NegativeLogLoss computes the negative log loss with respect to the targets.
func (m *Model) NegativeLogLoss(emissionScores []ag.DualValue, target []int) ag.DualValue {
	goldScore := m.goldScore(emissionScores, target)
	totalScore := m.totalScore(emissionScores)
	return ag.Sub(totalScore, goldScore)
}

func (m *Model) goldScore(emissionScores []ag.DualValue, target []int) ag.DualValue {
	goldScore := ag.At(emissionScores[0], target[0], 0)
	goldScore = ag.Add(goldScore, ag.At(m.TransitionScores, 0, target[0]+1)) // start transition
	prevIndex := target[0] + 1
	for i := 1; i < len(target); i++ {
		goldScore = ag.Add(goldScore, ag.At(emissionScores[i], target[i]))
		goldScore = ag.Add(goldScore, ag.At(m.TransitionScores, prevIndex, target[i]+1))
		prevIndex = target[i] + 1
	}
	goldScore = ag.Add(goldScore, ag.At(m.TransitionScores, prevIndex, 0)) // end transition
	return goldScore
}

func (m *Model) totalScore(predicted []ag.DualValue) ag.DualValue {
	totalVector := m.totalScoreStart(predicted[0])
	for i := 1; i < len(predicted); i++ {
		totalVector = m.totalScoreStep(totalVector, ag.SeparateVec(predicted[i]))
	}
	totalVector = m.totalScoreEnd(totalVector)
	return ag.Log(ag.ReduceSum(ag.Concat(totalVector...)))
}

func (m *Model) totalScoreStart(stepVec ag.DualValue) []ag.DualValue {
	scores := make([]ag.DualValue, m.Size)
	for i := 0; i < m.Size; i++ {
		scores[i] = ag.Add(ag.At(stepVec, i), ag.At(m.TransitionScores, 0, i+1))
	}
	return scores
}

func (m *Model) totalScoreEnd(stepVec []ag.DualValue) []ag.DualValue {
	scores := make([]ag.DualValue, m.Size)
	for i := 0; i < m.Size; i++ {
		vecTrans := ag.Add(stepVec[i], ag.At(m.TransitionScores, i+1, 0))
		scores[i] = ag.Add(scores[i], ag.Exp(vecTrans))
	}
	return scores
}

func (m *Model) totalScoreStep(totalVec []ag.DualValue, stepVec []ag.DualValue) []ag.DualValue {
	scores := make([]ag.DualValue, m.Size)
	for i := 0; i < m.Size; i++ {
		nodei := totalVec[i]
		for j := 0; j < m.Size; j++ {
			vecSum := ag.Add(nodei, stepVec[j])
			vecTrans := ag.Add(vecSum, ag.At(m.TransitionScores, i+1, j+1))
			scores[j] = ag.Add(scores[j], ag.Exp(vecTrans))
		}
	}
	for i := 0; i < m.Size; i++ {
		scores[i] = ag.Log(scores[i])
	}
	return scores
}
