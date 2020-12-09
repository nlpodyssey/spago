// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crf

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	TransitionScores *nn.Param `type:"weights"`
}

func New(labels int) *Model {
	return &Model{
		TransitionScores: nn.NewParam(mat.NewEmptyDense(labels+1, labels+1)), // +1 for start and end transitions
	}
}

func (m *Model) Predict(emissionScores []ag.Node) []int {
	return Viterbi(m.TransitionScores.Value(), emissionScores)
}

type Processor struct {
	nn.BaseProcessor
	size             int
	transitionScores [][]ag.Node
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
		size:             m.TransitionScores.Value().Rows() - 1,
		transitionScores: nn.Separate(ctx.Graph, ctx.Graph.NewWrap(m.TransitionScores)),
	}
}

// Forward is not available for the CRF. Use Predict() instead.
func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("crf: Forward() not available. Use Predict() instead.")
}

func (p *Processor) NegativeLogLoss(emissionScores []ag.Node, target []int) ag.Node {
	goldScore := p.goldScore(emissionScores, target)
	totalScore := p.totalScore(emissionScores)
	return p.Graph.Sub(totalScore, goldScore)
}

func (p *Processor) goldScore(emissionScores []ag.Node, target []int) ag.Node {
	g := p.Graph
	goldScore := g.At(emissionScores[0], target[0], 0)
	goldScore = g.Add(goldScore, p.transitionScores[0][target[0]+1]) // start transition
	prevIndex := target[0] + 1
	for i := 1; i < len(target); i++ {
		goldScore = g.Add(goldScore, g.AtVec(emissionScores[i], target[i]))
		goldScore = g.Add(goldScore, p.transitionScores[prevIndex][target[i]+1])
		prevIndex = target[i] + 1
	}
	goldScore = g.Add(goldScore, p.transitionScores[prevIndex][0]) // end transition
	return goldScore
}

func (p *Processor) totalScore(predicted []ag.Node) ag.Node {
	g := p.Graph
	totalVector := p.totalScoreStart(predicted[0])
	for i := 1; i < len(predicted); i++ {
		totalVector = p.totalScoreStep(totalVector, nn.SeparateVec(g, predicted[i]))
	}
	totalVector = p.totalScoreEnd(totalVector)
	return g.Log(g.ReduceSum(g.Concat(totalVector...)))

}

func (p *Processor) totalScoreStart(stepVec ag.Node) []ag.Node {
	firstTransitionScores := p.transitionScores[0]
	scores := make([]ag.Node, p.size)
	g := p.Graph
	for i := 0; i < p.size; i++ {
		scores[i] = g.Add(g.AtVec(stepVec, i), firstTransitionScores[i+1])
	}
	return scores
}

func (p *Processor) totalScoreEnd(stepVec []ag.Node) []ag.Node {
	scores := make([]ag.Node, p.size)
	g := p.Graph
	for i := 0; i < p.size; i++ {
		vecTrans := g.Add(stepVec[i], p.transitionScores[i+1][0])
		scores[i] = g.Add(scores[i], g.Exp(vecTrans))
	}
	return scores
}

func (p *Processor) totalScoreStep(totalVec []ag.Node, stepVec []ag.Node) []ag.Node {
	scores := make([]ag.Node, p.size)
	g := p.Graph
	for i := 0; i < p.size; i++ {
		nodei := totalVec[i]
		transitionScores := p.transitionScores[i+1]
		for j := 0; j < p.size; j++ {
			vecSum := g.Add(nodei, stepVec[j])
			vecTrans := g.Add(vecSum, transitionScores[j+1])
			scores[j] = g.Add(scores[j], g.Exp(vecTrans))
		}
	}
	for i := 0; i < p.size; i++ {
		scores[i] = g.Log(scores[i])
	}
	return scores
}
