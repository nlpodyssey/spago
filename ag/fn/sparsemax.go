// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"sort"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

// SparseMax function implementation, based on https://github.com/gokceneraslan/SparseMax.torch
type SparseMax[O Operand] struct {
	x        O
	y        mat.Matrix // initialized during the forward pass, required by the backward pass
	operands []O
}

// NewSparseMax returns a new SparseMax Function.
func NewSparseMax[O Operand](x O) *SparseMax[O] {
	return &SparseMax[O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *SparseMax[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *SparseMax[O]) Forward() mat.Matrix {
	x := r.x.Value()
	xMax := x.Max().Scalar().Float64()

	// translate the input by max for numerical stability
	v := x.SubScalar(xMax)

	zs, cumSumInput, _, tau := sparseMaxCommon(v)
	mat.ReleaseMatrix(zs)
	mat.ReleaseMatrix(cumSumInput)

	v.SubScalarInPlace(tau).ClipInPlace(0, xMax)

	r.y = v
	return v
}

// Backward computes the backward pass.
func (r *SparseMax[O]) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		var nzSum float64
		var nzCount float64
		r.y.DoVecNonZero(func(i int, _ float64) {
			nzSum += gy.ScalarAtVec(i).Float64()
			nzCount++
		})
		nzSum = nzSum / nzCount

		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		r.y.DoVecNonZero(func(i int, _ float64) {
			gyi := gy.ScalarAtVec(i).Float64()
			gx.SetVecScalar(i, float.Float(gyi-nzSum))
		})

		r.x.AccGrad(gx)
	}
}

func sparseMaxCommon(v mat.Matrix) (zs, cumSumInput mat.Matrix, bounds []float64, tau float64) {
	// FIXME: avoid casting to specific type
	zsData := make([]float64, v.Size())
	copy(zsData, v.Data().Float64())

	// Sort zs in descending order.
	sort.Slice(zsData, func(i, j int) bool {
		return zsData[i] > zsData[j]
	})

	zs = mat.NewVecDense(zsData)

	bounds = make([]float64, len(zsData))
	for i := range bounds {
		bounds[i] = 1 + float64(i+1)*zsData[i]
	}

	cumSumInput = zs.CumSum()
	cumSumInputData := cumSumInput.Data().Float64()

	k := -1
	tau = 0.0
	for i := range zsData {
		if bounds[i] > cumSumInputData[i] {
			if k < (i + 1) {
				k = i + 1
			}
			tau += zsData[i]
		}
	}
	tau = (tau - 1) / float64(k)

	return zs, cumSumInput, bounds, tau
}
