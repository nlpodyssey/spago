// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"sort"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

// SparseMax function implementation, based on https://github.com/gokceneraslan/SparseMax.torch
type SparseMax[O mat.Tensor] struct {
	x O
	y mat.Matrix // initialized during the forward pass, required by the backward pass
}

// NewSparseMax returns a new SparseMax Function.
func NewSparseMax[O mat.Tensor](x O) *SparseMax[O] {
	return &SparseMax[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *SparseMax[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the function.
func (r *SparseMax[O]) Forward() (mat.Tensor, error) {
	x := r.x.Value().(mat.Matrix)
	xMax := x.Max().Item().F64()

	// translate the input by max for numerical stability
	v := x.SubScalar(xMax)

	_, _, _, tau := sparseMaxCommon(v)

	v.SubScalarInPlace(tau).ClipInPlace(0, xMax)

	r.y = v
	return v, nil
}

// Backward computes the backward pass.
func (r *SparseMax[O]) Backward(gy mat.Tensor) error {
	if r.x.RequiresGrad() {
		var nzSum float64
		var nzCount float64
		r.y.DoVecNonZero(func(i int, _ float64) {
			nzSum += gy.(mat.Matrix).ScalarAt(i).F64()
			nzCount++
		})
		nzSum = nzSum / nzCount

		gx := r.x.Value().(mat.Matrix).ZerosLike()
		r.y.DoVecNonZero(func(i int, _ float64) {
			gyi := gy.(mat.Matrix).ScalarAt(i).F64()
			gx.SetScalar(float.Interface(gyi-nzSum), i)
		})

		r.x.AccGrad(gx)
	}
	return nil
}

func sparseMaxCommon(v mat.Matrix) (zs, cumSumInput mat.Matrix, bounds []float64, tau float64) {
	// FIXME: avoid casting to specific type
	zsData := make([]float64, v.Size())
	copy(zsData, v.Data().F64())

	// Sort zs in descending order.
	sort.Slice(zsData, func(i, j int) bool {
		return zsData[i] > zsData[j]
	})

	zs = mat.NewDense[float64](mat.WithBacking(zsData))

	bounds = make([]float64, len(zsData))
	for i := range bounds {
		bounds[i] = 1 + float64(i+1)*zsData[i]
	}

	cumSumInput = zs.CumSum()
	cumSumInputData := cumSumInput.Data().F64()

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
