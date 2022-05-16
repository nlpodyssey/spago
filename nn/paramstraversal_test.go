// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"reflect"
	"sync"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

type ParamsTraversalTester struct {
	CollectedParams []Param
}

func NewParamsTraversalTester() *ParamsTraversalTester {
	return &ParamsTraversalTester{
		CollectedParams: make([]Param, 0),
	}
}
func (ptt *ParamsTraversalTester) collect(param Param, _ string, _ ParamsType) {
	ptt.CollectedParams = append(ptt.CollectedParams, param)
}

// ParamsTraversalBaseModel can be used as base Model in tests.
// The sole purpose of this struct is to satisfy the Model interface,
// providing a fake Reify method.
type ParamsTraversalBaseModel struct {
	Module
}

var _ Model = &ParamsTraversalBaseModel{}

func (ParamsTraversalBaseModel) Forward(_ any) any {
	panic("this should never be called")
}

func TestParamsTraversal(t *testing.T) {
	t.Run("float32", testParamsTraversal[float32])
	t.Run("float64", testParamsTraversal[float64])
}

type emptyStruct struct{}

type ptModel1 struct {
	ParamsTraversalBaseModel
	a int
	b string
	c []int
	d []float32
	e []emptyStruct
	f map[bool]emptyStruct
}

type ptModel2 struct {
	ParamsTraversalBaseModel
	A Param
	B Param
}

type ptModel3 struct {
	ParamsTraversalBaseModel
	A []Param
	B []Param
}

type ptModel4 struct {
	ParamsTraversalBaseModel
	P Param
	M Model
}

type ptModel5 struct {
	ParamsTraversalBaseModel
	P Param
	M []Model
}

type ptModel6 struct {
	ParamsTraversalBaseModel
	P Param
	M []any
}

type testStructP struct {
	P Param
}

func (t testStructP) TraverseParams(callback ParamsTraversalFunc) {
	callback(t.P, "p", Weights)
}

type testStructQ struct {
	ParamsTraversalBaseModel
	P Param
}

type testStructToIgnore struct {
	P Param
}

type ptModel7 struct {
	ParamsTraversalBaseModel
	ToIgnore1 []testStructToIgnore
	ToIgnore2 testStructQ // ignored when strict exploration
	S         []testStructP
}

type ptModel8 struct {
	ParamsTraversalBaseModel
	MI map[int]Param
	MS map[string]Param
}

type ptModel9 struct {
	ParamsTraversalBaseModel
	MS testStructP
	MP *testStructP
}

type ptModel10 struct {
	ParamsTraversalBaseModel
	MS *sync.Map
}

func testParamsTraversal[T float.DType](t *testing.T) {
	t.Parallel()

	t.Run("empty model", func(t *testing.T) {
		t.Parallel()

		m := &ParamsTraversalBaseModel{}
		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		assertEqual(t, tt.CollectedParams, []Param{})
	})

	t.Run("irrelevant fields are ignored", func(t *testing.T) {
		t.Parallel()

		m := &ptModel1{
			a: 0,
			b: "",
			c: []int{1, 2},
			d: []float32{3.4, 5.6},
			e: []emptyStruct{{}, {}},
			f: map[bool]emptyStruct{true: {}, false: {}},
		}
		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		assertEqual(t, tt.CollectedParams, []Param{})
	})

	t.Run("it visits Param fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel2{
			A: NewParam(mat.NewScalar[T](1)),
			B: NewParam(mat.NewScalar[T](2)),
		}
		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param{m.A, m.B}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits []Param fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel3{
			A: []Param{
				NewParam(mat.NewScalar[T](1)),
				NewParam(mat.NewScalar[T](2)),
			},
			B: []Param{
				NewParam(mat.NewScalar[T](3)),
				NewParam(mat.NewScalar[T](4)),
			},
		}
		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param{m.A[0], m.A[1], m.B[0], m.B[1]}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it can optionally visit nested Model fields", func(t *testing.T) {
		t.Parallel()

		nestedModel := &ptModel4{
			P: NewParam(mat.NewScalar[T](100)),
			M: &ParamsTraversalBaseModel{},
		}

		m := &ptModel4{
			P: NewParam(mat.NewScalar[T](1)),
			M: nestedModel,
		}

		t.Run("with exploreSubModels false", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, false)
			pt.walk(m)

			expected := []Param{m.P}
			assertEqual(t, tt.CollectedParams, expected)
		})

		t.Run("with exploreSubModels true", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, true)
			pt.walk(m)

			expected := []Param{m.P, nestedModel.P}
			assertEqual(t, tt.CollectedParams, expected)
		})
	})

	t.Run("it can optionally visit nested []Model fields", func(t *testing.T) {
		t.Parallel()

		mA := &ptModel5{P: NewParam(mat.NewScalar[T](100))}
		mB := &ptModel5{P: NewParam(mat.NewScalar[T](200))}

		m := &ptModel5{
			P: NewParam(mat.NewScalar[T](1)),
			M: []Model{mA, mB},
		}

		t.Run("with exploreSubModels false", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, false)
			pt.walk(m)

			expected := []Param{m.P}
			assertEqual(t, tt.CollectedParams, expected)
		})

		t.Run("with exploreSubModels true", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, true)
			pt.walk(m)

			expected := []Param{m.P, mA.P, mB.P}
			assertEqual(t, tt.CollectedParams, expected)
		})
	})

	t.Run("it can optionally visit nested Model items in slice fields", func(t *testing.T) {
		t.Parallel()

		mA := &ptModel6{P: NewParam(mat.NewScalar[T](100))}
		mB := &ptModel6{P: NewParam(mat.NewScalar[T](200))}

		m := &ptModel6{
			P: NewParam(mat.NewScalar[T](1)),
			M: []any{mA, mB},
		}

		t.Run("with exploreSubModels false", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, false)
			pt.walk(m)

			expected := []Param{m.P}
			assertEqual(t, tt.CollectedParams, expected)
		})

		t.Run("with exploreSubModels true", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, true)
			pt.walk(m)

			expected := []Param{m.P, mA.P, mB.P}
			assertEqual(t, tt.CollectedParams, expected)
		})
	})

	t.Run("it visits struct items in params-annotated slice fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel7{
			ToIgnore1: []testStructToIgnore{
				{P: NewParam(mat.NewScalar[T](1))},
				{P: NewParam(mat.NewScalar[T](2))},
			},
			ToIgnore2: testStructQ{
				P: NewParam(mat.NewScalar[T](1)),
			},
			S: []testStructP{
				{P: NewParam(mat.NewScalar[T](10))},
				{P: NewParam(mat.NewScalar[T](20))},
			},
		}

		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param{m.S[0].P, m.S[1].P}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits Param items in map[int] and map[string] fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel8{
			MI: map[int]Param{
				0: NewParam(mat.NewScalar[T](1)),
			},
			MS: map[string]Param{
				"a": NewParam(mat.NewScalar[T](3)),
			},
		}

		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param{m.MI[0], m.MS["a"]}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits Param items in params-annotated struct of ptr fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel9{
			MS: testStructP{P: NewParam(mat.NewScalar[T](1))},
			MP: &testStructP{P: NewParam(mat.NewScalar[T](2))},
		}

		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param{m.MS.P, m.MP.P}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits Param items in params-annotated sync.Map fields", func(t *testing.T) {
		t.Parallel()

		m := &ptModel10{
			MS: &sync.Map{},
		}
		m.MS.Store("a", NewParam(mat.NewScalar[T](3)))

		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		p, _ := m.MS.Load("a")
		expected := []Param{p.(Param)}
		assertEqual(t, tt.CollectedParams, expected)
	})
}

func assertEqual(t *testing.T, actual, expected any) {
	t.Helper()
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected\n  %#v\nactual\n  %#v", expected, actual)
	}
}
