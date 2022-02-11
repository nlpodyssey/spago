// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings/syncmap"
	"reflect"
	"sync"
	"testing"
)

type ParamsTraversalTester struct {
	CollectedParams []Param[mat.Float]
}

func NewParamsTraversalTester() *ParamsTraversalTester {
	return &ParamsTraversalTester{CollectedParams: make([]Param[mat.Float], 0)}
}

func (ptt *ParamsTraversalTester) collect(param Param[mat.Float]) {
	ptt.CollectedParams = append(ptt.CollectedParams, param)
}

// ParamsTraversalBaseModel can be used as base Model in tests.
// The sole purpose of this struct is to satisfy the Model interface,
// providing a fake Reify method.
type ParamsTraversalBaseModel struct {
	BaseModel[mat.Float]
}

var _ Model[mat.Float] = &ParamsTraversalBaseModel{}

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

		assertEqual(t, tt.CollectedParams, []Param[mat.Float]{})
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

		assertEqual(t, tt.CollectedParams, []Param[mat.Float]{})
	})

	t.Run("it visits Param fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ParamsTraversalBaseModel
			A Param[mat.Float]
			B Param[mat.Float]
		}

		m := &TestModel{
			A: NewParam[mat.Float](mat.NewScalar[mat.Float](1)),
			B: NewParam[mat.Float](mat.NewScalar[mat.Float](2)),
		}
		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param[mat.Float]{m.A, m.B}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits []Param fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ParamsTraversalBaseModel
			A []Param[mat.Float]
			B []Param[mat.Float]
		}

		m := &TestModel{
			A: []Param[mat.Float]{
				NewParam[mat.Float](mat.NewScalar[mat.Float](1)),
				NewParam[mat.Float](mat.NewScalar[mat.Float](2)),
			},
			B: []Param[mat.Float]{
				NewParam[mat.Float](mat.NewScalar[mat.Float](3)),
				NewParam[mat.Float](mat.NewScalar[mat.Float](4)),
			},
		}
		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param[mat.Float]{m.A[0], m.A[1], m.B[0], m.B[1]}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it can optionally visit nested Model fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ParamsTraversalBaseModel
			P Param[mat.Float]
			M Model[mat.Float]
		}

		nestedModel := &TestModel{
			P: NewParam[mat.Float](mat.NewScalar[mat.Float](100)),
			M: &ParamsTraversalBaseModel{},
		}

		m := &TestModel{
			P: NewParam[mat.Float](mat.NewScalar[mat.Float](1)),
			M: nestedModel,
		}

		t.Run("with exploreSubModels false", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, false)
			pt.walk(m)

			expected := []Param[mat.Float]{m.P}
			assertEqual(t, tt.CollectedParams, expected)
		})

		t.Run("with exploreSubModels true", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, true)
			pt.walk(m)

			expected := []Param[mat.Float]{m.P, nestedModel.P}
			assertEqual(t, tt.CollectedParams, expected)
		})
	})

	t.Run("it can optionally visit nested []Model fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ParamsTraversalBaseModel
			P Param[mat.Float]
			M []Model[mat.Float]
		}

		mA := &TestModel{P: NewParam[mat.Float](mat.NewScalar[mat.Float](100))}
		mB := &TestModel{P: NewParam[mat.Float](mat.NewScalar[mat.Float](200))}

		m := &TestModel{
			P: NewParam[mat.Float](mat.NewScalar[mat.Float](1)),
			M: []Model[mat.Float]{mA, mB},
		}

		t.Run("with exploreSubModels false", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, false)
			pt.walk(m)

			expected := []Param[mat.Float]{m.P}
			assertEqual(t, tt.CollectedParams, expected)
		})

		t.Run("with exploreSubModels true", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, true)
			pt.walk(m)

			expected := []Param[mat.Float]{m.P, mA.P, mB.P}
			assertEqual(t, tt.CollectedParams, expected)
		})
	})

	t.Run("it can optionally visit nested Model items in slice fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ParamsTraversalBaseModel
			P Param[mat.Float]
			M []interface{}
		}

		mA := &TestModel{P: NewParam[mat.Float](mat.NewScalar[mat.Float](100))}
		mB := &TestModel{P: NewParam[mat.Float](mat.NewScalar[mat.Float](200))}

		m := &TestModel{
			P: NewParam[mat.Float](mat.NewScalar[mat.Float](1)),
			M: []interface{}{mA, mB},
		}

		t.Run("with exploreSubModels false", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, false)
			pt.walk(m)

			expected := []Param[mat.Float]{m.P}
			assertEqual(t, tt.CollectedParams, expected)
		})

		t.Run("with exploreSubModels true", func(t *testing.T) {
			tt := NewParamsTraversalTester()

			pt := newParamsTraversal(tt.collect, true)
			pt.walk(m)

			expected := []Param[mat.Float]{m.P, mA.P, mB.P}
			assertEqual(t, tt.CollectedParams, expected)
		})
	})

	t.Run("it visits struct items in params-annotated slice fields", func(t *testing.T) {
		t.Parallel()

		type MyStruct struct {
			P Param[mat.Float]
		}

		type TestModel struct {
			ParamsTraversalBaseModel
			Ignored []MyStruct
			S       []MyStruct `spago:"type:params"`
		}

		m := &TestModel{
			Ignored: []MyStruct{
				{P: NewParam[mat.Float](mat.NewScalar[mat.Float](1))},
				{P: NewParam[mat.Float](mat.NewScalar[mat.Float](2))},
			},
			S: []MyStruct{
				{P: NewParam[mat.Float](mat.NewScalar[mat.Float](10))},
				{P: NewParam[mat.Float](mat.NewScalar[mat.Float](20))},
			},
		}

		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param[mat.Float]{m.S[0].P, m.S[1].P}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits Param items in map[int] and map[string] fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ParamsTraversalBaseModel
			MI map[int]Param[mat.Float]
			MS map[string]Param[mat.Float]
		}

		m := &TestModel{
			MI: map[int]Param[mat.Float]{
				0: NewParam[mat.Float](mat.NewScalar[mat.Float](1)),
			},
			MS: map[string]Param[mat.Float]{
				"a": NewParam[mat.Float](mat.NewScalar[mat.Float](3)),
			},
		}

		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param[mat.Float]{m.MI[0], m.MS["a"]}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits Param items in params-annotated struct of ptr fields", func(t *testing.T) {
		t.Parallel()

		type MyStruct struct {
			P Param[mat.Float]
		}

		type TestModel struct {
			ParamsTraversalBaseModel
			MS MyStruct  `spago:"type:params"`
			MP *MyStruct `spago:"type:params"`
		}

		m := &TestModel{
			MS: MyStruct{P: NewParam[mat.Float](mat.NewScalar[mat.Float](1))},
			MP: &MyStruct{P: NewParam[mat.Float](mat.NewScalar[mat.Float](2))},
		}

		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		expected := []Param[mat.Float]{m.MS.P, m.MP.P}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits Param items in params-annotated sync.Map fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ParamsTraversalBaseModel
			MS *sync.Map `spago:"type:params"`
		}

		m := &TestModel{
			MS: &sync.Map{},
		}
		m.MS.Store("a", NewParam[mat.Float](mat.NewScalar[mat.Float](3)))

		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		p, _ := m.MS.Load("a")
		expected := []Param[mat.Float]{p.(Param[mat.Float])}
		assertEqual(t, tt.CollectedParams, expected)
	})

	t.Run("it visits Param items in params-annotated embeddings.syncmap.Map fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ParamsTraversalBaseModel
			MS *syncmap.Map `spago:"type:params"`
		}

		m := &TestModel{
			MS: syncmap.New(),
		}
		m.MS.Store("a", NewParam[mat.Float](mat.NewScalar[mat.Float](3)))

		tt := NewParamsTraversalTester()

		pt := newParamsTraversal(tt.collect, false)
		pt.walk(m)

		p, _ := m.MS.Load("a")
		expected := []Param[mat.Float]{p.(Param[mat.Float])}
		assertEqual(t, tt.CollectedParams, expected)
	})
}

func assertEqual(t *testing.T, actual, expected interface{}) {
	t.Helper()
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected\n  %#v\nactual\n  %#v", expected, actual)
	}
}
