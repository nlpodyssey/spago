// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"math"
	"sort"
)

// SparseMax function implementation, based on https://github.com/gokceneraslan/SparseMax.torch
type SparseMax struct {
	x Operand
	y mat.Matrix // initialized during the forward pass, required by the backward pass
}

var _ Function = &SparseMax{}
var _ Function = &SparseMaxLoss{}

func NewSparseMax(x Operand) *SparseMax {
	return &SparseMax{x: x}
}

func (s *SparseMax) Forward() mat.Matrix {
	s.y = mat.NewVecDense(sparseMax(translateInput(s.x.Value().Data())))
	return s.y
}

func (s *SparseMax) Backward(gy mat.Matrix) {
	if s.x.RequiresGrad() {
		output := s.y.Data()
		nzSum := 0.0
		nzCount := 0.0
		gx := mat.GetDenseWorkspace(s.x.Value().Rows(), s.x.Value().Columns())
		defer mat.ReleaseDense(gx)
		for i := range output {
			if output[i] != 0 {
				nzSum += gy.At(i, 0)
				nzCount++
			}
		}
		nzSum = nzSum / nzCount

		for i := range output {
			if output[i] != 0 {
				gx.Set(i, 0, gy.At(i, 0)-nzSum)
			} else {
				gx.Set(i, 0, 0)
			}
		}
		s.x.PropagateGrad(gx)
	}
}

// translateInput translates the input by max for numerical stability
func translateInput(v []float64) []float64 {
	maximum := max(v)
	translated := make([]float64, len(v))
	for i := range v {
		translated[i] = v[i] - maximum
	}
	return translated
}

func sparseMaxCommon(v []float64) (zs []float64, bounds []float64, cumSumInput []float64, tau float64) {
	zs = make([]float64, len(v))
	copy(zs, v)

	// Sort zs in descending order.
	sort.Sort(sort.Reverse(sort.Float64Slice(zs)))

	bounds = make([]float64, len(zs))
	for i := range bounds {
		bounds[i] = 1 + float64(i+1)*zs[i]
	}

	cumSumInput = make([]float64, len(zs))
	f64utils.CumSum(cumSumInput, zs)

	k := -1
	tau = 0.0
	for i := range zs {
		if bounds[i] > cumSumInput[i] {
			if k < (i + 1) {
				k = i + 1
			}
			tau += zs[i]
		}
	}
	tau = (tau - 1) / float64(k)

	return zs, bounds, cumSumInput, tau
}

func sparseMax(v []float64) []float64 {
	zs, _, _, tau := sparseMaxCommon(v)

	//Reuses zs to avoid allocating new slice
	for i := range zs {
		zs[i] = math.Max(0.0, v[i]-tau)
	}
	return zs
}

type SparseMaxLoss struct {
	x   Operand
	tau float64    // computed during the forward pass
	y   mat.Matrix // computed during forward pass
}

func NewSparseMaxLoss(x Operand) *SparseMaxLoss {
	return &SparseMaxLoss{x: x}
}

// sparseMaxLoss computes the sparseMax loss function and returns
// the loss and the tau parameter (needed by backward)
func sparseMaxLoss(v []float64) ([]float64, float64) {
	zs, bounds, cumSumInput, tau := sparseMaxCommon(v)
	regTerm := 0.0
	tauSquared := tau * tau

	for i := range zs {
		if bounds[i] > cumSumInput[i] {
			regTerm += zs[i]*zs[i] - tauSquared
		}
	}
	regTerm = regTerm*0.5 + 0.5

	// Reuse zs to avoid allocating a new slice
	for i := range zs {
		zs[i] = v[i] - regTerm
	}
	return zs, tau
}
func (s *SparseMaxLoss) Forward() mat.Matrix {
	output, tau := sparseMaxLoss(s.x.Value().Data())
	s.y = mat.NewVecDense(output)
	s.tau = tau
	return s.y
}

func (s *SparseMaxLoss) Backward(gy mat.Matrix) {
	if s.x.RequiresGrad() {
		input := s.x.Value().Data()
		sparseMax := make([]float64, len(input))
		for i := range sparseMax {
			sparseMax[i] = math.Max(0, input[i]-s.tau)
		}
		gx := mat.GetDenseWorkspace(s.x.Value().Rows(), s.x.Value().Columns())
		defer mat.ReleaseDense(gx)
		gyData := gy.Data()
		gySum := f64utils.Sum(gyData)
		for i := range gyData {
			gx.Set(i, 0, gy.At(i, 0)-gySum*sparseMax[i])
		}
		s.x.PropagateGrad(gx)
	}
}
