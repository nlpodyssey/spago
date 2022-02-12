// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crf

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/matutils"
)

// ViterbiStructure implements Viterbi decoding.
type ViterbiStructure[T mat.DType] struct {
	scores       mat.Matrix[T]
	backpointers []int
}

// NewViterbiStructure returns a new ViterbiStructure ready to use.
func NewViterbiStructure[T mat.DType](size int) *ViterbiStructure[T] {
	return &ViterbiStructure[T]{
		scores:       mat.NewInitVecDense(size, mat.Inf[T](-1)),
		backpointers: make([]int, size),
	}
}

// Viterbi decodes the xs sequence according to the transitionMatrix.
func Viterbi[T mat.DType](transitionMatrix mat.Matrix[T], xs []ag.Node[T]) []int {
	alpha := make([]*ViterbiStructure[T], len(xs)+1)
	alpha[0] = viterbiStepStart(transitionMatrix, xs[0].Value())
	for i := 1; i < len(xs); i++ {
		alpha[i] = viterbiStep(transitionMatrix, alpha[i-1].scores, xs[i].Value())
	}
	alpha[len(xs)] = viterbiStepEnd(transitionMatrix, alpha[len(xs)-1].scores)

	ys := make([]int, len(xs))
	ys[len(xs)-1] = matutils.ArgMax(alpha[len(xs)].scores.Data())
	for i := len(xs) - 2; i >= 0; i-- {
		ys[i] = alpha[i+1].backpointers[ys[i+1]]
	}
	return ys
}

func viterbiStepStart[T mat.DType](transitionMatrix, maxVec mat.Matrix[T]) *ViterbiStructure[T] {
	y := NewViterbiStructure[T](transitionMatrix.Rows() - 1)
	for i := 0; i < transitionMatrix.Rows()-1; i++ {
		score := maxVec.At(i, 0) + transitionMatrix.At(0, i+1)
		if score > y.scores.At(i, 0) {
			y.scores.SetVec(i, score)
			y.backpointers[i] = i
		}
	}
	return y
}

func viterbiStepEnd[T mat.DType](transitionMatrix, maxVec mat.Matrix[T]) *ViterbiStructure[T] {
	y := NewViterbiStructure[T](transitionMatrix.Rows() - 1)
	for i := 0; i < transitionMatrix.Rows()-1; i++ {
		score := maxVec.At(i, 0) + transitionMatrix.At(i+1, 0)
		if score > y.scores.At(i, 0) {
			y.scores.SetVec(i, score)
			y.backpointers[i] = i
		}
	}
	return y
}

func viterbiStep[T mat.DType](transitionMatrix, maxVec, stepVec mat.Matrix[T]) *ViterbiStructure[T] {
	y := NewViterbiStructure[T](transitionMatrix.Rows() - 1)
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
