// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/mattest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ Function[*variable] = &Affine[*variable]{}

func TestAffine(t *testing.T) {
	t.Run("float32", testAffine[float32])
	t.Run("float64", testAffine[float64])
}

func testAffine[T float.DType](t *testing.T) {
	for _, lenWXs := range []int{1, 3, 5} {
		t.Run(fmt.Sprintf("it panics if len(wxs) is %d", lenWXs), func(t *testing.T) {
			b := newVarWithGrad(mat.NewScalar(T(42)))
			wxs := make([]*variable, lenWXs)
			for i := range wxs {
				wxs[i] = newVarWithGrad(mat.NewScalar(T(i)))
			}
			require.Panics(t, func() { NewAffine(b, wxs...) })
		})
	}

	tests := []struct {
		name        string
		b           mat.Matrix
		wxs         []mat.Matrix
		wantFwd     mat.Matrix
		gy          mat.Matrix
		wantBGrad   mat.Matrix
		wantWXGrads []mat.Matrix // w1, x1, w2, x2, ... wn, xn
	}{
		{
			name: "one (w, x) pair - scalars",
			b:    mat.NewScalar(T(10)),
			wxs: []mat.Matrix{
				mat.NewScalar(T(2)), // w
				mat.NewScalar(T(3)), // x
			},
			wantFwd:   mat.NewScalar(T(16)),
			gy:        mat.NewScalar(T(4)),
			wantBGrad: mat.NewScalar(T(4)),
			wantWXGrads: []mat.Matrix{
				mat.NewScalar(T(12)), // w
				mat.NewScalar(T(8)),  // x
			},
		},
		{
			name: "two (w, x) pairs - scalars",
			b:    mat.NewScalar(T(10)),
			wxs: []mat.Matrix{
				mat.NewScalar(T(2)),  // w1
				mat.NewScalar(T(3)),  // x1
				mat.NewScalar(T(10)), // w2
				mat.NewScalar(T(20)), // x2
			},
			wantFwd:   mat.NewScalar(T(216)),
			gy:        mat.NewScalar(T(4)),
			wantBGrad: mat.NewScalar(T(4)),
			wantWXGrads: []mat.Matrix{
				mat.NewScalar(T(12)), // w1
				mat.NewScalar(T(8)),  // x1
				mat.NewScalar(T(80)), // w2
				mat.NewScalar(T(40)), // x2
			},
		},
		{
			name: "one (w, x) pair - w matrix, x vector",
			b:    mat.NewVecDense([]T{2, 3}),
			wxs: []mat.Matrix{
				mat.NewDense(2, 2, []T{ // w
					4, 5,
					6, 7,
				}),
				mat.NewVecDense([]T{8, 9}), // x
			},
			wantFwd:   mat.NewVecDense([]T{79, 114}),
			gy:        mat.NewVecDense([]T{10, 11}),
			wantBGrad: mat.NewVecDense([]T{10, 11}),
			wantWXGrads: []mat.Matrix{
				mat.NewDense(2, 2, []T{ // w
					80, 90,
					88, 99,
				}),
				mat.NewVecDense([]T{106, 127}), // x
			},
		},
		{
			name: "two (w, x) pairs - w matrices, x vectors",
			b:    mat.NewVecDense([]T{2, 3}),
			wxs: []mat.Matrix{
				mat.NewDense(2, 2, []T{ // w1
					4, 5,
					6, 7,
				}),
				mat.NewVecDense([]T{8, 9}), // x1
				mat.NewDense(2, 2, []T{ // w2
					10, 11,
					12, 13,
				}),
				mat.NewVecDense([]T{14, 15}), // x2
			},
			wantFwd:   mat.NewVecDense([]T{384, 477}),
			gy:        mat.NewVecDense([]T{16, 17}),
			wantBGrad: mat.NewVecDense([]T{16, 17}),
			wantWXGrads: []mat.Matrix{
				mat.NewDense(2, 2, []T{ // w1
					128, 144,
					136, 153,
				}),
				mat.NewVecDense([]T{166, 199}), // x1
				mat.NewDense(2, 2, []T{ // w2
					224, 240,
					238, 255,
				}),
				mat.NewVecDense([]T{364, 397}), // x2
			},
		},
		{
			name: "one (w, x) pair - w and x matrices",
			b: mat.NewDense(3, 2, []T{
				2, 3,
				4, 5,
				6, 7,
			}),
			wxs: []mat.Matrix{
				mat.NewDense(3, 2, []T{ // w
					8, 9,
					10, 11,
					12, 13,
				}),
				mat.NewDense(2, 2, []T{ // x
					14, 15,
					16, 17,
				}),
			},
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
			wantWXGrads: []mat.Matrix{
				mat.NewDense(3, 2, []T{ // w
					537, 611,
					595, 677,
					653, 743,
				}),
				mat.NewDense(2, 2, []T{ // x
					608, 638,
					668, 701,
				}),
			},
		},
		{
			name: "(w, x) pairs where x is nil are ignored",
			b:    mat.NewScalar(T(10)),
			wxs: []mat.Matrix{
				mat.NewScalar(T(2)),   // w1
				mat.NewScalar(T(3)),   // x1
				mat.NewScalar(T(987)), // w2
				nil,                   // x2
				mat.NewScalar(T(10)),  // w3
				mat.NewScalar(T(20)),  // x3
			},
			wantFwd:   mat.NewScalar(T(216)),
			gy:        mat.NewScalar(T(4)),
			wantBGrad: mat.NewScalar(T(4)),
			wantWXGrads: []mat.Matrix{
				mat.NewScalar(T(12)), // w1
				mat.NewScalar(T(8)),  // x1
				nil,                  // w2
				nil,                  // x2
				mat.NewScalar(T(80)), // w3
				mat.NewScalar(T(40)), // x3
			},
		},
		{
			name:        "no (w, x) pairs",
			b:           mat.NewScalar(T(42)),
			wxs:         []mat.Matrix{},
			wantFwd:     mat.NewScalar(T(42)),
			gy:          mat.NewScalar(T(4)),
			wantBGrad:   mat.NewScalar(T(4)),
			wantWXGrads: []mat.Matrix{},
		},
		{
			name: "(w, x) pairs with all x values set to nil",
			b:    mat.NewScalar(T(42)),
			wxs: []mat.Matrix{
				mat.NewScalar(T(123)), // w1
				nil,                   // x1
				mat.NewScalar(T(456)), // w2
				nil,                   // x2
			},
			wantFwd:   mat.NewScalar(T(42)),
			gy:        mat.NewScalar(T(4)),
			wantBGrad: mat.NewScalar(T(4)),
			wantWXGrads: []mat.Matrix{
				nil, // w1
				nil, // x1
				nil, // w2
				nil, // x2
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Len(t, tt.wantWXGrads, len(tt.wxs), "malformed test case")

			b := newVarWithGrad(tt.b)
			wxs := make([]*variable, len(tt.wxs))
			for i, v := range tt.wxs {
				if v == nil {
					continue
				}
				wxs[i] = newVarWithGrad(v)
			}

			f := NewAffine(b, wxs...)
			y := f.Forward()
			mattest.RequireMatrixEquals(t, tt.wantFwd, y)

			f.Backward(tt.gy)
			mattest.AssertMatrixEquals(t, tt.wantBGrad, b.grad, "bias grad")
			for i, want := range tt.wantWXGrads {
				if want == nil {
					if wxs[i] != nil {
						assert.Nilf(t, wxs[i].grad, "wxs[%d] grad", i)
					}
					continue
				}
				mattest.AssertMatrixEquals(t, want, wxs[i].grad, "wxs[", i, "] grad")
			}
		})
	}
}
