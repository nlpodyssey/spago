// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"sort"
)

// SparseMax function implementation, based on https://github.com/gokceneraslan/SparseMax.torch
type SparseMax[T mat.DType, O Operand[T]] struct {
	x        O
	y        mat.Matrix // initialized during the forward pass, required by the backward pass
	operands []O
}

// NewSparseMax returns a new SparseMax Function.
func NewSparseMax[T mat.DType, O Operand[T]](x O) *SparseMax[T, O] {
	return &SparseMax[T, O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *SparseMax[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *SparseMax[T, O]) Forward() mat.Matrix {
	x := r.x.Value()
	xMax := x.Max().Scalar().Float64()

	// translate the input by max for numerical stability
	v := x.SubScalar(xMax)

	zs, cumSumInput, _, tau := sparseMaxCommon[T](v)
	mat.ReleaseMatrix(zs)
	mat.ReleaseMatrix(cumSumInput)

	v.SubScalarInPlace(float64(tau)).ClipInPlace(0, xMax)

	r.y = v
	return v
}

// Backward computes the backward pass.
func (r *SparseMax[T, O]) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		var nzSum T = 0.0
		var nzCount T = 0.0
		r.y.DoVecNonZero(func(i int, _ float64) {
			nzSum += mat.DTFloat[T](gy.ScalarAtVec(i))
			nzCount++
		})
		nzSum = nzSum / nzCount

		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		r.y.DoVecNonZero(func(i int, _ float64) {
			gyi := mat.DTFloat[T](gy.ScalarAtVec(i))
			gx.SetVecScalar(i, mat.Float(gyi-nzSum))
		})

		r.x.AccGrad(gx)
	}
}

func sparseMaxCommon[T mat.DType](v mat.Matrix) (zs, cumSumInput mat.Matrix, bounds []T, tau T) {
	zsData := make([]T, v.Size())
	copy(zsData, mat.Data[T](v))

	// Sort zs in descending order.
	sort.Slice(zsData, func(i, j int) bool {
		return zsData[i] > zsData[j]
	})

	zs = mat.NewVecDense(zsData)

	bounds = make([]T, len(zsData))
	for i := range bounds {
		bounds[i] = 1 + T(i+1)*zsData[i]
	}

	cumSumInput = zs.CumSum()
	cumSumInputData := mat.Data[T](cumSumInput)

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
	tau = (tau - 1) / T(k)

	return zs, cumSumInput, bounds, tau
}
