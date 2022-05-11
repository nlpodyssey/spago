// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"reflect"
	"sync"
	"testing"

	"github.com/nlpodyssey/spago/mat"
)

type ParamsTraversalTester[T mat.DType] struct {
	CollectedParams []Param[T]
}

func NewParamsTraversalTester[T mat.DType]() *ParamsTraversalTester[T] {
	return &ParamsTraversalTester[T]{
		CollectedParams: make([]Param[T], 0),
	}
}
func (ptt *ParamsTraversalTester[T]) collect(param Param[T], _ string, _ ParamsType) {
	ptt.CollectedParams = append(ptt.CollectedParams, param)
}

// ParamsTraversalBaseModel can be used as base Model in tests.
// The sole purpose of this struct is to satisfy the Model interface,
// providing a fake Reify method.
type ParamsTraversalBaseModel[T mat.DType] struct {
	Module[T]
}

var _ Model[float32] = &ParamsTraversalBaseModel[float32]{}

func (ParamsTraversalBaseModel[_]) Forward(_ any) any {
	panic("this should never be called")
}

func TestParamsTraversal(t *testing.T) {
	t.Run("float32", testParamsTraversal[float32])
	t.Run("float64", testParamsTraversal[float64])
}

type emptyStruct struct{}

type ptModel1[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	a int
	b string
	c []int
	d []float32
	e []emptyStruct
	f map[bool]emptyStruct
}

type ptModel2[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	A Param[T]
	B Param[T]
}

type ptModel3[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	A []Param[T]
	B []Param[T]
}

type ptModel4[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	P Param[T]
	M Model[T]
}

type ptModel5[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	P Param[T]
	M []Model[T]
}

type ptModel6[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	P Param[T]
	M []any
}

type testStructP[T mat.DType] struct {
	P Param[T]
}

func (t testStructP[T]) TraverseParams(callback ParamsTraversalFunc[T]) {
	callback(t.P, "p", Weights)
}

type testStructQ[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	P Param[T]
}

type testStructToIgnore[T mat.DType] struct {
	P Param[T]
}

type ptModel7[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	ToIgnore1 []testStructToIgnore[T]
	ToIgnore2 testStructQ[T] // ignored when strict exploration
	S         []testStructP[T]
}

type ptModel8[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	MI map[int]Param[T]
	MS map[string]Param[T]
}

type ptModel9[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	MS testStructP[T]
	MP *testStructP[T]
}

type ptModel10[T mat.DType] struct {
	ParamsTraversalBaseModel[T]
	MS *sync.Map
}

func testParamsTraversal[T mat.DType](t *testing.T) {
	t.Parallel()

	t.Run("empty model", func(t *testing.T) {
		t.Parallel()

		m := &ParamsTraversalBaseModel[T]{}
		tt := NewParamsTraversalTester[T]()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		assertEqual(t, tt.CollectedParams, []Param[T]{})
	})

	t.Run("irrelevant fields are ignored", func(t *testing.T) {
		t.Parallel()

		m := &ptModel1[T]{
			a: 0,
			b: "",
			c: []int{1, 2},
			d: []float32{3.4, 5.6},
			e: []emptyStruct{{}, {}},
			f: map[bool]emptyStruct{true: {}, false: {}},
		}
		tt := NewParamsTraversalTester[T]()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		assertEqual(t, tt.CollectedParams, []Param[T]{})
	})

	t.Run("it visits Param fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel2[T]{
			A: NewParam[T](mat.NewScalar[T](1)),
			B: NewParam[T](mat.NewScalar[T](2)),
		}
		tt := NewParamsTraversalTester[T]()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param[T]{m.A, m.B}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits []Param fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel3[T]{
			A: []Param[T]{
				NewParam[T](mat.NewScalar[T](1)),
				NewParam[T](mat.NewScalar[T](2)),
			},
			B: []Param[T]{
				NewParam[T](mat.NewScalar[T](3)),
				NewParam[T](mat.NewScalar[T](4)),
			},
		}
		tt := NewParamsTraversalTester[T]()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param[T]{m.A[0], m.A[1], m.B[0], m.B[1]}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it can optionally visit nested Model fields", func(t *testing.T) {
		t.Parallel()

		nestedModel := &ptModel4[T]{
			P: NewParam[T](mat.NewScalar[T](100)),
			M: &ParamsTraversalBaseModel[T]{},
		}

		m := &ptModel4[T]{
			P: NewParam[T](mat.NewScalar[T](1)),
			M: nestedModel,
		}

		t.Run("with exploreSubModels false", func(t *testing.T) {
			tt := NewParamsTraversalTester[T]()

			pt := newParamsTraversal(tt.collect, false)
			pt.walk(m)

			expected := []Param[T]{m.P}
			assertEqual(t, tt.CollectedParams, expected)
		})

		t.Run("with exploreSubModels true", func(t *testing.T) {
			tt := NewParamsTraversalTester[T]()

			pt := newParamsTraversal(tt.collect, true)
			pt.walk(m)

			expected := []Param[T]{m.P, nestedModel.P}
			assertEqual(t, tt.CollectedParams, expected)
		})
	})

	t.Run("it can optionally visit nested []Model fields", func(t *testing.T) {
		t.Parallel()

		mA := &ptModel5[T]{P: NewParam[T](mat.NewScalar[T](100))}
		mB := &ptModel5[T]{P: NewParam[T](mat.NewScalar[T](200))}

		m := &ptModel5[T]{
			P: NewParam[T](mat.NewScalar[T](1)),
			M: []Model[T]{mA, mB},
		}

		t.Run("with exploreSubModels false", func(t *testing.T) {
			tt := NewParamsTraversalTester[T]()

			pt := newParamsTraversal(tt.collect, false)
			pt.walk(m)

			expected := []Param[T]{m.P}
			assertEqual(t, tt.CollectedParams, expected)
		})

		t.Run("with exploreSubModels true", func(t *testing.T) {
			tt := NewParamsTraversalTester[T]()

			pt := newParamsTraversal(tt.collect, true)
			pt.walk(m)

			expected := []Param[T]{m.P, mA.P, mB.P}
			assertEqual(t, tt.CollectedParams, expected)
		})
	})

	t.Run("it can optionally visit nested Model items in slice fields", func(t *testing.T) {
		t.Parallel()

		mA := &ptModel6[T]{P: NewParam[T](mat.NewScalar[T](100))}
		mB := &ptModel6[T]{P: NewParam[T](mat.NewScalar[T](200))}

		m := &ptModel6[T]{
			P: NewParam[T](mat.NewScalar[T](1)),
			M: []any{mA, mB},
		}

		t.Run("with exploreSubModels false", func(t *testing.T) {
			tt := NewParamsTraversalTester[T]()

			pt := newParamsTraversal(tt.collect, false)
			pt.walk(m)

			expected := []Param[T]{m.P}
			assertEqual(t, tt.CollectedParams, expected)
		})

		t.Run("with exploreSubModels true", func(t *testing.T) {
			tt := NewParamsTraversalTester[T]()

			pt := newParamsTraversal(tt.collect, true)
			pt.walk(m)

			expected := []Param[T]{m.P, mA.P, mB.P}
			assertEqual(t, tt.CollectedParams, expected)
		})
	})

	t.Run("it visits struct items in params-annotated slice fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel7[T]{
			ToIgnore1: []testStructToIgnore[T]{
				{P: NewParam[T](mat.NewScalar[T](1))},
				{P: NewParam[T](mat.NewScalar[T](2))},
			},
			ToIgnore2: testStructQ[T]{
				P: NewParam[T](mat.NewScalar[T](1)),
			},
			S: []testStructP[T]{
				{P: NewParam[T](mat.NewScalar[T](10))},
				{P: NewParam[T](mat.NewScalar[T](20))},
			},
		}

		tt := NewParamsTraversalTester[T]()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param[T]{m.S[0].P, m.S[1].P}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits Param items in map[int] and map[string] fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel8[T]{
			MI: map[int]Param[T]{
				0: NewParam[T](mat.NewScalar[T](1)),
			},
			MS: map[string]Param[T]{
				"a": NewParam[T](mat.NewScalar[T](3)),
			},
		}

		tt := NewParamsTraversalTester[T]()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param[T]{m.MI[0], m.MS["a"]}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits Param items in params-annotated struct of ptr fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel9[T]{
			MS: testStructP[T]{P: NewParam[T](mat.NewScalar[T](1))},
			MP: &testStructP[T]{P: NewParam[T](mat.NewScalar[T](2))},
		}

		tt := NewParamsTraversalTester[T]()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param[T]{m.MS.P, m.MP.P}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits Param items in params-annotated sync.Map fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel10[T]{
			MS: &sync.Map{},
		}
		m.MS.Store("a", NewParam[T](mat.NewScalar[T](3)))

		tt := NewParamsTraversalTester[T]()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		p, _ := m.MS.Load("a")
		expected := []Param[T]{p.(Param[T])}
		assertEqual(t, tt.CollectedParams, expected)
	})
}

func assertEqual(t *testing.T, actual, expected any) {
	t.Helper()
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected\n  %#v\nactual\n  %#v", expected, actual)
	}
}
