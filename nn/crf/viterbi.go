// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crf

import (
	"math"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

// FIXME: ViterbiStructure currently works with float64 only

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
func Viterbi(transitionMatrix mat.Matrix, xs []ag.DualValue) []int {
	alpha := make([]*ViterbiStructure, len(xs)+1)
	alpha[0] = viterbiStepStart(transitionMatrix, xs[0].Value())
	for i := 1; i < len(xs); i++ {
		alpha[i] = viterbiStep(transitionMatrix, alpha[i-1].scores, xs[i].Value())
	}
	alpha[len(xs)] = viterbiStepEnd(transitionMatrix, alpha[len(xs)-1].scores)

	ys := make([]int, len(xs))
	ys[len(xs)-1] = alpha[len(xs)].scores.ArgMax()
	for i := len(xs) - 2; i >= 0; i-- {
		ys[i] = alpha[i+1].backpointers[ys[i+1]]
	}
	return ys
}

func viterbiStepStart(transitionMatrix, maxVec mat.Matrix) *ViterbiStructure {
	y := NewViterbiStructure(transitionMatrix.Rows() - 1)
	for i := 0; i < transitionMatrix.Rows()-1; i++ {
		mv := maxVec.ScalarAt(i, 0).F64()
		tv := transitionMatrix.ScalarAt(0, i+1).F64()
		yv := y.scores.ScalarAt(i, 0).F64()
		score := mv + tv
		if score > yv {
			y.scores.SetScalar(float.Interface(score), i)
			y.backpointers[i] = i
		}
	}
	return y
}

func viterbiStepEnd(transitionMatrix, maxVec mat.Matrix) *ViterbiStructure {
	y := NewViterbiStructure(transitionMatrix.Rows() - 1)
	for i := 0; i < transitionMatrix.Rows()-1; i++ {
		mv := maxVec.ScalarAt(i, 0).F64()
		tv := transitionMatrix.ScalarAt(i+1, 0).F64()
		yv := y.scores.ScalarAt(i, 0).F64()
		score := mv + tv
		if score > yv {
			y.scores.SetScalar(float.Interface(score), i)
			y.backpointers[i] = i
		}
	}
	return y
}

func viterbiStep(transitionMatrix, maxVec, stepVec mat.Matrix) *ViterbiStructure {
	y := NewViterbiStructure(transitionMatrix.Rows() - 1)
	for i := 0; i < transitionMatrix.Rows()-1; i++ {
		for j := 0; j < transitionMatrix.Cols()-1; j++ {
			mv := maxVec.ScalarAt(i, 0).F64()
			sv := stepVec.ScalarAt(j, 0).F64()
			tv := transitionMatrix.ScalarAt(i+1, j+1).F64()
			yv := y.scores.ScalarAt(j, 0).F64()
			score := mv + sv + tv
			if score > yv {
				y.scores.SetScalar(float.Interface(score), j)
				y.backpointers[j] = i
			}
		}
	}
	return y
}
