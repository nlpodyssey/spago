// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	matsort "github.com/nlpodyssey/spago/pkg/mat32/sort"
	"sort"
)

// SparseMax function implementation, based on https://github.com/gokceneraslan/SparseMax.torch
type SparseMax struct {
	x Operand
	y mat.Matrix // initialized during the forward pass, required by the backward pass
}

var _ Function = &SparseMax{}
var _ Function = &SparseMaxLoss{}

// NewSparseMax returns a new SparseMax Function.
func NewSparseMax(x Operand) *SparseMax {
	return &SparseMax{x: x}
}

// Forward computes the output of the function.
func (s *SparseMax) Forward() mat.Matrix {
	s.y = mat.NewVecDense(sparseMax(translateInput(s.x.Value().Data())))
	return s.y
}

// Backward computes the backward pass.
func (s *SparseMax) Backward(gy mat.Matrix) {
	if s.x.RequiresGrad() {
		output := s.y.Data()
		var nzSum mat.Float = 0.0
		var nzCount mat.Float = 0.0
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
func translateInput(v []mat.Float) []mat.Float {
	maximum := max(v)
	translated := make([]mat.Float, len(v))
	for i := range v {
		translated[i] = v[i] - maximum
	}
	return translated
}

func sparseMaxCommon(v []mat.Float) (zs []mat.Float, bounds []mat.Float, cumSumInput []mat.Float, tau mat.Float) {
	zs = make([]mat.Float, len(v))
	copy(zs, v)

	// Sort zs in descending order.
	sort.Sort(sort.Reverse(matsort.FloatSlice(zs)))

	bounds = make([]mat.Float, len(zs))
	for i := range bounds {
		bounds[i] = 1 + mat.Float(i+1)*zs[i]
	}

	cumSumInput = make([]mat.Float, len(zs))
	floatutils.CumSum(cumSumInput, zs)

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
	tau = (tau - 1) / mat.Float(k)

	return zs, bounds, cumSumInput, tau
}

func sparseMax(v []mat.Float) []mat.Float {
	zs, _, _, tau := sparseMaxCommon(v)

	//Reuses zs to avoid allocating new slice
	for i := range zs {
		zs[i] = mat.Max(0.0, v[i]-tau)
	}
	return zs
}

// SparseMaxLoss function implementation, based on https://github.com/gokceneraslan/SparseMax.torch
type SparseMaxLoss struct {
	x   Operand
	tau mat.Float  // computed during the forward pass
	y   mat.Matrix // computed during forward pass
}

// NewSparseMaxLoss returns a new SparseMaxLoss Function.
func NewSparseMaxLoss(x Operand) *SparseMaxLoss {
	return &SparseMaxLoss{x: x}
}

// sparseMaxLoss computes the sparseMax loss function and returns
// the loss and the tau parameter (needed by backward)
func sparseMaxLoss(v []mat.Float) ([]mat.Float, mat.Float) {
	zs, bounds, cumSumInput, tau := sparseMaxCommon(v)
	var regTerm mat.Float = 0.0
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

// Forward computes the output of the function.
func (s *SparseMaxLoss) Forward() mat.Matrix {
	output, tau := sparseMaxLoss(s.x.Value().Data())
	s.y = mat.NewVecDense(output)
	s.tau = tau
	return s.y
}

// Backward computes the backward pass.
func (s *SparseMaxLoss) Backward(gy mat.Matrix) {
	if s.x.RequiresGrad() {
		input := s.x.Value().Data()
		sparseMax := make([]mat.Float, len(input))
		for i := range sparseMax {
			sparseMax[i] = mat.Max(0, input[i]-s.tau)
		}
		gx := mat.GetDenseWorkspace(s.x.Value().Rows(), s.x.Value().Columns())
		defer mat.ReleaseDense(gx)
		gyData := gy.Data()
		gySum := floatutils.Sum(gyData)
		for i := range gyData {
			gx.Set(i, 0, gy.At(i, 0)-gySum*sparseMax[i])
		}
		s.x.PropagateGrad(gx)
	}
}
