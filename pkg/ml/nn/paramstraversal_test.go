// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"reflect"
	"testing"
)

type ParamsTraversalTester struct {
	CollectedParams []Param
}

func NewParamsTraversalTester() *ParamsTraversalTester {
	return &ParamsTraversalTester{CollectedParams: make([]Param, 0)}
}

func (ptt *ParamsTraversalTester) collect(param Param) {
	ptt.CollectedParams = append(ptt.CollectedParams, param)
}

// ParamsTraversalBaseModel can be used as base Model in tests.
// The sole purpose of this struct is to satisfy the Model interface,
// providing a fake Reify method.
type ParamsTraversalBaseModel struct {
	BaseModel
}

var _ Model = &ParamsTraversalBaseModel{}

func (p ParamsTraversalBaseModel) Forward(_ interface{}) interface{} {
	panic("this should never be called")
}

func TestParamsTraversal(t *testing.T) {
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

		type MyStruct struct{}

		type TestModel struct {
			ParamsTraversalBaseModel
			a int
			b string
			c []int
			d []float32
			e []MyStruct
			f map[bool]MyStruct
		}
		m := &TestModel{
			c: []int{1, 2},
			d: []float32{3.4, 5.6},
			e: []MyStruct{{}, {}},
			f: map[bool]MyStruct{true: {}, false: {}},
		}
		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		assertEqual(t, tt.CollectedParams, []Param{})
	})

	t.Run("it visits Param fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ParamsTraversalBaseModel
			A Param
			B Param
		}

		m := &TestModel{
			A: NewParam(mat.NewScalar(1)),
			B: NewParam(mat.NewScalar(2)),
		}
		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param{m.A, m.B}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits []Param fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ParamsTraversalBaseModel
			A []Param
			B []Param
		}

		m := &TestModel{
			A: []Param{
				NewParam(mat.NewScalar(1)),
				NewParam(mat.NewScalar(2)),
			},
			B: []Param{
				NewParam(mat.NewScalar(3)),
				NewParam(mat.NewScalar(4)),
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

		type TestModel struct {
			ParamsTraversalBaseModel
			P Param
			M Model
		}

		nestedModel := &TestModel{
			P: NewParam(mat.NewScalar(100)),
			M: &ParamsTraversalBaseModel{},
		}

		m := &TestModel{
			P: NewParam(mat.NewScalar(1)),
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

		type TestModel struct {
			ParamsTraversalBaseModel
			P Param
			M []Model
		}

		mA := &TestModel{P: NewParam(mat.NewScalar(100))}
		mB := &TestModel{P: NewParam(mat.NewScalar(200))}

		m := &TestModel{
			P: NewParam(mat.NewScalar(1)),
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

		type TestModel struct {
			ParamsTraversalBaseModel
			P Param
			M []interface{}
		}

		mA := &TestModel{P: NewParam(mat.NewScalar(100))}
		mB := &TestModel{P: NewParam(mat.NewScalar(200))}

		m := &TestModel{
			P: NewParam(mat.NewScalar(1)),
			M: []interface{}{mA, mB},
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

		type MyStruct struct {
			P Param
		}

		type TestModel struct {
			ParamsTraversalBaseModel
			Ignored []MyStruct
			S       []MyStruct `spago:"type:params"`
		}

		m := &TestModel{
			Ignored: []MyStruct{
				{P: NewParam(mat.NewScalar(1))},
				{P: NewParam(mat.NewScalar(2))},
			},
			S: []MyStruct{
				{P: NewParam(mat.NewScalar(10))},
				{P: NewParam(mat.NewScalar(20))},
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

		type TestModel struct {
			ParamsTraversalBaseModel
			MI map[int]Param
			MS map[string]Param
		}

		m := &TestModel{
			MI: map[int]Param{
				0: NewParam(mat.NewScalar(1)),
			},
			MS: map[string]Param{
				"a": NewParam(mat.NewScalar(3)),
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

		type MyStruct struct {
			P Param
		}

		type TestModel struct {
			ParamsTraversalBaseModel
			MS MyStruct  `spago:"type:params"`
			MP *MyStruct `spago:"type:params"`
		}

		m := &TestModel{
			MS: MyStruct{P: NewParam(mat.NewScalar(1))},
			MP: &MyStruct{P: NewParam(mat.NewScalar(2))},
		}

		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param{m.MS.P, m.MP.P}
		assertEqual(t, tt.CollectedParams, expected)
	})
}

func assertEqual(t *testing.T, actual, expected interface{}) {
	t.Helper()
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected\n  %#v\nactual\n  %#v", expected, actual)
	}
}
