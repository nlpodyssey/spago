// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAffine(t *testing.T) {
	t.Run("float32", testAffine[float32])
	t.Run("float64", testAffine[float64])
}

func testAffine[T float.DType](t *testing.T) {
	for _, lenWXPairs := range []int{1, 3, 5} {
		t.Run(fmt.Sprintf("it panics if len(wxPairs) is %d", lenWXPairs), func(t *testing.T) {
			b := newVarWithGrad(mat.NewScalar(T(1)))
			w1 := newVarWithGrad(mat.NewScalar(T(2)))
			x1 := newVarWithGrad(mat.NewScalar(T(3)))
			wxPairs := make([]*variable, lenWXPairs)
			for i := range wxPairs {
				wxPairs[i] = newVarWithGrad(mat.NewScalar(T(i)))
			}
			require.Panics(t, func() { NewAffine(b, w1, x1, wxPairs...) })
		})
	}

	tests := []struct {
		name             string
		b                mat.Matrix
		w1               mat.Matrix
		x1               mat.Matrix
		wxPairs          []mat.Matrix
		wantFwd          mat.Matrix
		gy               mat.Matrix
		wantBGrad        mat.Matrix
		wantW1Grad       mat.Matrix
		wantX1Grad       mat.Matrix
		wantWXPairsGrads []mat.Matrix // w2, x2, ... wN, xN
	}{
		{
			name:             "no additional (w, x) pairs - scalars",
			b:                mat.NewScalar(T(10)),
			w1:               mat.NewScalar(T(2)),
			x1:               mat.NewScalar(T(3)),
			wxPairs:          nil,
			wantFwd:          mat.NewScalar(T(16)),
			gy:               mat.NewScalar(T(4)),
			wantBGrad:        mat.NewScalar(T(4)),
			wantW1Grad:       mat.NewScalar(T(12)),
			wantX1Grad:       mat.NewScalar(T(8)),
			wantWXPairsGrads: nil,
		},
		{
			name: "one additional (w, x) pair - scalars",
			b:    mat.NewScalar(T(10)),
			w1:   mat.NewScalar(T(2)),
			x1:   mat.NewScalar(T(3)),
			wxPairs: []mat.Matrix{
				mat.NewScalar(T(10)), // w2
				mat.NewScalar(T(20)), // x2
			},
			wantFwd:    mat.NewScalar(T(216)),
			gy:         mat.NewScalar(T(4)),
			wantBGrad:  mat.NewScalar(T(4)),
			wantW1Grad: mat.NewScalar(T(12)),
			wantX1Grad: mat.NewScalar(T(8)),
			wantWXPairsGrads: []mat.Matrix{
				mat.NewScalar(T(80)), // w2
				mat.NewScalar(T(40)), // x2
			},
		},
		{
			name: "no additional (w, x) pairs - w matrix, x vector",
			b:    mat.NewVecDense([]T{2, 3}),
			w1: mat.NewDense(2, 2, []T{
				4, 5,
				6, 7,
			}),
			x1:        mat.NewVecDense([]T{8, 9}),
			wxPairs:   nil,
			wantFwd:   mat.NewVecDense([]T{79, 114}),
			gy:        mat.NewVecDense([]T{10, 11}),
			wantBGrad: mat.NewVecDense([]T{10, 11}),
			wantW1Grad: mat.NewDense(2, 2, []T{
				80, 90,
				88, 99,
			}),
			wantX1Grad:       mat.NewVecDense([]T{106, 127}),
			wantWXPairsGrads: nil,
		},
		{
			name: "one additional (w, x) pair - w matrices, x vectors",
			b:    mat.NewVecDense([]T{2, 3}),
			w1: mat.NewDense(2, 2, []T{
				4, 5,
				6, 7,
			}),
			x1: mat.NewVecDense([]T{8, 9}),
			wxPairs: []mat.Matrix{
				mat.NewDense(2, 2, []T{ // w2
					10, 11,
					12, 13,
				}),
				mat.NewVecDense([]T{14, 15}), // x2
			},
			wantFwd:   mat.NewVecDense([]T{384, 477}),
			gy:        mat.NewVecDense([]T{16, 17}),
			wantBGrad: mat.NewVecDense([]T{16, 17}),
			wantW1Grad: mat.NewDense(2, 2, []T{
				128, 144,
				136, 153,
			}),
			wantX1Grad: mat.NewVecDense([]T{166, 199}),
			wantWXPairsGrads: []mat.Matrix{
				mat.NewDense(2, 2, []T{ // w2
					224, 240,
					238, 255,
				}),
				mat.NewVecDense([]T{364, 397}), // x2
			},
		},
		{
			name: "no additional (w, x) pairs - w and x matrices",
			b: mat.NewDense(3, 2, []T{
				2, 3,
				4, 5,
				6, 7,
			}),
			w1: mat.NewDense(3, 2, []T{
				8, 9,
				10, 11,
				12, 13,
			}),
			x1: mat.NewDense(2, 2, []T{
				14, 15,
				16, 17,
			}),
			wxPairs: nil,
			wantFwd: mat.NewDense(3, 2, []T{
				258, 276,
				320, 342,
				382, 408,
			}),
			gy: mat.NewDense(3, 2, []T{
				18, 19,
				20, 21,
				22, 23,
			}),
			wantBGrad: mat.NewDense(3, 2, []T{
				18, 19,
				20, 21,
				22, 23,
			}),
			wantW1Grad: mat.NewDense(3, 2, []T{
				537, 611,
				595, 677,
				653, 743,
			}),
			wantX1Grad: mat.NewDense(2, 2, []T{
				608, 638,
				668, 701,
			}),
			wantWXPairsGrads: nil,
		},
		{
			name: "additional (w, x) pairs where x is nil are ignored",
			b:    mat.NewScalar(T(10)),
			w1:   mat.NewScalar(T(2)),
			x1:   mat.NewScalar(T(3)),
			wxPairs: []mat.Matrix{
				mat.NewScalar(T(987)), // w2
				nil,                   // x2
				mat.NewScalar(T(10)),  // w3
				mat.NewScalar(T(20)),  // x3
			},
			wantFwd:    mat.NewScalar(T(216)),
			gy:         mat.NewScalar(T(4)),
			wantBGrad:  mat.NewScalar(T(4)),
			wantW1Grad: mat.NewScalar(T(12)),
			wantX1Grad: mat.NewScalar(T(8)),
			wantWXPairsGrads: []mat.Matrix{
				nil,                  // w2
				nil,                  // x2
				mat.NewScalar(T(80)), // w3
				mat.NewScalar(T(40)), // x3
			},
		},
		{
			name: "(w, x) pairs with all x values set to nil",
			b:    mat.NewScalar(T(10)),
			w1:   mat.NewScalar(T(2)),
			x1:   mat.NewScalar(T(3)),
			wxPairs: []mat.Matrix{
				mat.NewScalar(T(123)), // w2
				nil,                   // x2
				mat.NewScalar(T(456)), // w3
				nil,                   // x4
			},
			wantFwd:    mat.NewScalar(T(16)),
			gy:         mat.NewScalar(T(4)),
			wantBGrad:  mat.NewScalar(T(4)),
			wantW1Grad: mat.NewScalar(T(12)),
			wantX1Grad: mat.NewScalar(T(8)),
			wantWXPairsGrads: []mat.Matrix{
				nil, // w2
				nil, // x2
				nil, // w3
				nil, // x4
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Len(t, tt.wantWXPairsGrads, len(tt.wxPairs), "malformed test case")

			b := newVarWithGrad(tt.b)
			w1 := newVarWithGrad(tt.w1)
			x1 := newVarWithGrad(tt.x1)
			wxPairs := make([]*variable, len(tt.wxPairs))
			for i, v := range tt.wxPairs {
				if v == nil {
					continue
				}
				wxPairs[i] = newVarWithGrad(v)
			}

			f := NewAffine(b, w1, x1, wxPairs...)
			y, err := f.Forward()
			assert.Nil(t, err)
			mat.RequireMatrixEquals(t, tt.wantFwd, y)

			err = f.Backward(tt.gy)
			assert.Nil(t, err)
			mat.AssertMatrixEquals(t, tt.wantBGrad, b.grad, "bias grad")
			mat.AssertMatrixEquals(t, tt.wantW1Grad, w1.grad, "w1 grad")
			mat.AssertMatrixEquals(t, tt.wantX1Grad, x1.grad, "x1 grad")
			for i, want := range tt.wantWXPairsGrads {
				if want == nil {
					if wxPairs[i] != nil {
						assert.Nilf(t, wxPairs[i].grad, "wxPairs[%d] grad", i)
					}
					continue
				}
				mat.AssertMatrixEquals(t, want, wxPairs[i].grad, "wxPairs[", i, "] grad")
			}
		})
	}
}
