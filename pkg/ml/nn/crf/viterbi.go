// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crf

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"math"
)

// ViterbiStructure implements Viterbi decoding.
type ViterbiStructure struct {
	scores       mat.Matrix
	backpointers []int
}

// NewViterbiStructure returns a new ViterbiStructure ready to use.
func NewViterbiStructure(size int) *ViterbiStructure {
	return &ViterbiStructure{
		scores:       mat.NewInitVecDense(size, math.Inf(-1)),
		backpointers: make([]int, size),
	}
}

// Viterbi decodes the xs sequence according to the transitionMatrix.
func Viterbi(transitionMatrix mat.Matrix, xs []ag.Node) []int {
	alpha := make([]*ViterbiStructure, len(xs)+1)
	alpha[0] = ViterbiStepStart(transitionMatrix, xs[0].Value())
	for i := 1; i < len(xs); i++ {
		alpha[i] = ViterbiStep(transitionMatrix, alpha[i-1].scores, xs[i].Value())
	}
	alpha[len(xs)] = ViterbiStepEnd(transitionMatrix, alpha[len(xs)-1].scores)

	ys := make([]int, len(xs))
	ys[len(xs)-1] = f64utils.ArgMax(alpha[len(xs)].scores.Data())
	for i := len(xs) - 2; i >= 0; i-- {
		ys[i] = alpha[i+1].backpointers[ys[i+1]]
	}
	return ys
}

func ViterbiStepStart(transitionMatrix mat.Matrix, maxVec mat.Matrix) *ViterbiStructure {
	y := NewViterbiStructure(transitionMatrix.Rows() - 1)
	for i := 0; i < transitionMatrix.Rows()-1; i++ {
		score := maxVec.At(i, 0) + transitionMatrix.At(0, i+1)
		if score > y.scores.At(i, 0) {
			y.scores.SetVec(i, score)
			y.backpointers[i] = i
		}
	}
	return y
}

func ViterbiStepEnd(transitionMatrix mat.Matrix, maxVec mat.Matrix) *ViterbiStructure {
	y := NewViterbiStructure(transitionMatrix.Rows() - 1)
	for i := 0; i < transitionMatrix.Rows()-1; i++ {
		score := maxVec.At(i, 0) + transitionMatrix.At(i+1, 0)
		if score > y.scores.At(i, 0) {
			y.scores.SetVec(i, score)
			y.backpointers[i] = i
		}
	}
	return y
}

func ViterbiStep(transitionMatrix mat.Matrix, maxVec mat.Matrix, stepVec mat.Matrix) *ViterbiStructure {
	y := NewViterbiStructure(transitionMatrix.Rows() - 1)
	for i := 0; i < transitionMatrix.Rows()-1; i++ {
		for j := 0; j < transitionMatrix.Columns()-1; j++ {
			score := maxVec.At(i, 0) + stepVec.At(j, 0) + transitionMatrix.At(i+1, j+1)
			if score > y.scores.At(j, 0) {
				y.scores.SetVec(j, score)
				y.backpointers[j] = i
			}
		}
	}
	return y
}
