// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	matsort "github.com/nlpodyssey/spago/mat/sort"
	"sort"
)

// SparseMax function implementation, based on https://github.com/gokceneraslan/SparseMax.torch
type SparseMax[T mat.DType] struct {
	x Operand[T]
	y mat.Matrix[T] // initialized during the forward pass, required by the backward pass
}

var _ Function[float32] = &SparseMax[float32]{}
var _ Function[float32] = &SparseMaxLoss[float32]{}

// NewSparseMax returns a new SparseMax Function.
func NewSparseMax[T mat.DType](x Operand[T]) *SparseMax[T] {
	return &SparseMax[T]{x: x}
}

// Forward computes the output of the function.
func (s *SparseMax[T]) Forward() mat.Matrix[T] {
	x := s.x.Value()
	xMax := x.Max()

	// translate the input by max for numerical stability
	v := x.SubScalar(xMax)

	zs, cumSumInput, _, tau := sparseMaxCommon(v)
	mat.ReleaseMatrix(zs)
	mat.ReleaseMatrix(cumSumInput)

	v.SubScalarInPlace(tau).ClipInPlace(0, xMax)

	s.y = v
	return v
}

// Backward computes the backward pass.
func (s *SparseMax[T]) Backward(gy mat.Matrix[T]) {
	if s.x.RequiresGrad() {
		var nzSum T = 0.0
		var nzCount T = 0.0
		s.y.DoVecNonZero(func(i int, _ T) {
			nzSum += gy.AtVec(i)
			nzCount++
		})
		nzSum = nzSum / nzCount

		gx := s.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		s.y.DoVecNonZero(func(i int, _ T) {
			gx.SetVec(i, gy.AtVec(i)-nzSum)
		})

		s.x.PropagateGrad(gx)
	}
}

// SparseMaxLoss function implementation, based on https://github.com/gokceneraslan/SparseMax.torch
type SparseMaxLoss[T mat.DType] struct {
	x   Operand[T]
	tau T             // computed during the forward pass
	y   mat.Matrix[T] // computed during forward pass
}

// NewSparseMaxLoss returns a new SparseMaxLoss Function.
func NewSparseMaxLoss[T mat.DType](x Operand[T]) *SparseMaxLoss[T] {
	return &SparseMaxLoss[T]{x: x}
}

// Forward computes the output of the function.
func (s *SparseMaxLoss[T]) Forward() mat.Matrix[T] {
	v := s.x.Value().Clone()

	zs, cumSumInput, bounds, tau := sparseMaxCommon(v)
	defer mat.ReleaseMatrix(zs)
	defer mat.ReleaseMatrix(cumSumInput)

	tauSquared := tau * tau
	cumSumInputData := cumSumInput.Data()

	var regTerm T = 0.0
	for i, zsv := range zs.Data() {
		if bounds[i] > cumSumInputData[i] {
			regTerm += zsv*zsv - tauSquared
		}
	}

	regTerm = regTerm*0.5 + 0.5
	v.SubScalarInPlace(regTerm)

	s.y = v
	s.tau = tau
	return v
}

// Backward computes the backward pass.
func (s *SparseMaxLoss[T]) Backward(gy mat.Matrix[T]) {
	if s.x.RequiresGrad() {
		tau := s.tau
		gySum := gy.Sum()

		sparseMax := s.x.Value().Apply(func(_, _ int, v T) T {
			return mat.Max(0, v-tau) * gySum
		})
		defer mat.ReleaseMatrix(sparseMax)

		gx := gy.Sub(sparseMax)
		defer mat.ReleaseMatrix(gx)

		s.x.PropagateGrad(gx)
	}
}

func sparseMaxCommon[T mat.DType](v mat.Matrix[T]) (zs, cumSumInput mat.Matrix[T], bounds []T, tau T) {
	zsData := make([]T, v.Size())
	copy(zsData, v.Data())

	// Sort zs in descending order.
	sort.Sort(sort.Reverse(matsort.DTSlice[T](zsData)))
	zs = mat.NewVecDense(zsData)

	bounds = make([]T, len(zsData))
	for i := range bounds {
		bounds[i] = 1 + T(i+1)*zsData[i]
	}

	cumSumInput = zs.VecCumSum()
	cumSumInputData := cumSumInput.Data()

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
