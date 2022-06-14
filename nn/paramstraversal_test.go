// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"sync"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

type collectedParam struct {
	param Param
	name  string
	pType ParamsType
}

type collectedModel struct {
	model Model
	name  string
}

type traversalTest struct {
	name                 string
	model                Model
	subMod               bool
	expectedParams       []collectedParam
	expectedParamsStrict []collectedParam
	expectedModels       []collectedModel
}

type traversableType struct {
	fn func(ParamsTraversalFunc)
}

var _ ParamsTraverser = traversableType{}

func (t traversableType) TraverseParams(f ParamsTraversalFunc) {
	t.fn(f)
}

var traversalTests = []traversalTest{
	func() traversalTest {
		name := "empty model"

		type modelType struct {
			Module
		}

		m := &modelType{}

		return traversalTest{
			name:                 name,
			model:                m,
			subMod:               false,
			expectedParams:       nil,
			expectedParamsStrict: nil,
			expectedModels:       []collectedModel{{m, ""}},
		}
	}(),
	func() traversalTest {
		name := "irrelevant fields"

		type modelType struct {
			Module
			a int
			b string
			c []int
			d []float32
			e []struct{}
			f map[bool]struct{}
		}

		m := &modelType{
			a: 0,
			b: "",
			c: []int{1, 2},
			d: []float32{3.4, 5.6},
			e: []struct{}{{}, {}},
			f: map[bool]struct{}{true: {}, false: {}},
		}

		return traversalTest{
			name:                 name,
			model:                m,
			subMod:               false,
			expectedParams:       nil,
			expectedParamsStrict: nil,
			expectedModels:       []collectedModel{{m, ""}},
		}
	}(),
	func() traversalTest {
		name := "Param fields"

		type modelType struct {
			Module
			A Param
			B Param
		}

		m := &modelType{
			A: NewParam(mat.NewScalar(1.)),
			B: NewParam(mat.NewScalar(2.)),
		}

		params := []collectedParam{
			{m.A, "A", Undefined},
			{m.B, "B", Undefined},
		}
		return traversalTest{
			name:                 name,
			model:                m,
			subMod:               false,
			expectedParams:       params,
			expectedParamsStrict: params,
			expectedModels:       []collectedModel{{m, ""}},
		}
	}(),
	func() traversalTest {
		name := "[]Param fields"

		type modelType struct {
			Module
			A []Param
			B []Param
		}

		m := &modelType{
			A: []Param{
				NewParam(mat.NewScalar(1.)),
				NewParam(mat.NewScalar(2.)),
			},
			B: []Param{
				NewParam(mat.NewScalar(3.)),
				NewParam(mat.NewScalar(4.)),
			},
		}

		params := []collectedParam{
			{m.A[0], "A", Undefined},
			{m.A[1], "A", Undefined},
			{m.B[0], "B", Undefined},
			{m.B[1], "B", Undefined},
		}
		return traversalTest{
			name:                 name,
			model:                m,
			subMod:               false,
			expectedParams:       params,
			expectedParamsStrict: params,
			expectedModels:       []collectedModel{{m, ""}},
		}
	}(),
	func() traversalTest {
		name := "nested Model fields"

		type modelType struct {
			Module
			P Param
			M Model
		}

		type emptyModel struct {
			Module
		}

		nested := &modelType{
			P: NewParam(mat.NewScalar(100.)),
			M: &emptyModel{},
		}
		m := &modelType{
			P: NewParam(mat.NewScalar(1.)),
			M: nested,
		}

		return traversalTest{
			name:   name,
			model:  m,
			subMod: true,
			expectedParams: []collectedParam{
				{m.P, "P", Undefined},
				{nested.P, "P", Undefined},
			},
			expectedParamsStrict: []collectedParam{
				{m.P, "P", Undefined},
			},
			expectedModels: []collectedModel{
				{m, ""},
				{nested, "M"},
				{nested.M, "M"},
			},
		}
	}(),
	func() traversalTest {
		name := "nested []Model fields"

		type modelType struct {
			Module
			P Param
			M []Model
		}

		mA := &modelType{P: NewParam(mat.NewScalar(100.))}
		mB := &modelType{P: NewParam(mat.NewScalar(200.))}
		m := &modelType{
			P: NewParam(mat.NewScalar(1.)),
			M: []Model{mA, mB},
		}

		return traversalTest{
			name:   name,
			model:  m,
			subMod: true,
			expectedParams: []collectedParam{
				{m.P, "P", Undefined},
				{mA.P, "P", Undefined},
				{mB.P, "P", Undefined},
			},
			expectedParamsStrict: []collectedParam{
				{m.P, "P", Undefined},
			},
			expectedModels: []collectedModel{
				{m, ""},
				{mA, "M"},
				{mB, "M"},
			},
		}
	}(),
	func() traversalTest {
		name := "nested Model items in slice fields"

		type modelType struct {
			Module
			P Param
			M []any
		}

		mA := &modelType{P: NewParam(mat.NewScalar(100.))}
		mB := &modelType{P: NewParam(mat.NewScalar(200.))}
		m := &modelType{
			P: NewParam(mat.NewScalar(1.)),
			M: []any{mA, mB},
		}

		return traversalTest{
			name:   name,
			model:  m,
			subMod: true,
			expectedParams: []collectedParam{
				{m.P, "P", Undefined},
				{mA.P, "P", Undefined},
				{mB.P, "P", Undefined},
			},
			expectedParamsStrict: []collectedParam{
				{m.P, "P", Undefined},
			},
			expectedModels: []collectedModel{
				{m, ""},
				{mA, "M"},
				{mB, "M"},
			},
		}
	}(),
	func() traversalTest {
		name := "fields with type implementing ParamsTraverser"

		type simpleStruct struct {
			P Param
		}

		type modelType struct {
			Module
			Foo Param
			Bar []simpleStruct
			Baz Model
			Qux []traversableType
		}

		alfa := NewParam(mat.NewScalar(10.))
		bravo := NewParam(mat.NewScalar(20.))
		charlie := NewParam(mat.NewScalar(30.))
		delta := NewParam(mat.NewScalar(40.))
		echo := NewParam(mat.NewScalar(50.))

		nested := &modelType{
			Foo: NewParam(mat.NewScalar(3.)),
			Bar: nil,
			Baz: nil,
			Qux: []traversableType{
				{func(f ParamsTraversalFunc) {
					f(delta, "Delta", Undefined)
					f(echo, "", Biases)
				}},
			},
		}
		m := &modelType{
			Foo: NewParam(mat.NewScalar(1.)),
			Bar: []simpleStruct{{P: NewParam(mat.NewScalar(2.))}},
			Baz: nested,
			Qux: []traversableType{
				{func(f ParamsTraversalFunc) {
					f(alfa, "Alfa", Biases)
					f(bravo, "Bravo", Weights)
				}},
				{func(f ParamsTraversalFunc) {
					f(charlie, "Charlie", Undefined)
				}},
			},
		}

		return traversalTest{
			name:   name,
			model:  m,
			subMod: true,
			expectedParams: []collectedParam{
				{m.Foo, "Foo", Undefined},
				{nested.Foo, "Foo", Undefined},
				{delta, "Delta", Undefined},
				{echo, "", Biases},
				{alfa, "Alfa", Biases},
				{bravo, "Bravo", Weights},
				{charlie, "Charlie", Undefined},
			},
			expectedParamsStrict: []collectedParam{
				{m.Foo, "Foo", Undefined},
				{alfa, "Alfa", Biases},
				{bravo, "Bravo", Weights},
				{charlie, "Charlie", Undefined},
			},
			expectedModels: []collectedModel{
				{m, ""},
				{nested, "Baz"},
			},
		}
	}(),
	func() traversalTest {
		name := "map[int] and map[string] fields"

		type modelType struct {
			Module
			MI map[int]Param
			MS map[string]Param
		}

		m := &modelType{
			MI: map[int]Param{
				0: NewParam(mat.NewScalar(1.)),
			},
			MS: map[string]Param{
				"a": NewParam(mat.NewScalar(3.)),
			},
		}

		params := []collectedParam{
			{m.MI[0], "MI.0", Undefined},
			{m.MS["a"], "MS.a", Undefined},
		}
		return traversalTest{
			name:                 name,
			model:                m,
			subMod:               true,
			expectedParams:       params,
			expectedParamsStrict: params,
			expectedModels:       []collectedModel{{m, ""}},
		}
	}(),
	func() traversalTest {
		name := "sync.Map fields"

		type modelType struct {
			Module
			StrMap  *sync.Map
			IntMap  *sync.Map
			UintMap *sync.Map
		}

		foo := NewParam(mat.NewScalar(1.))
		bar := NewParam(mat.NewScalar(2.))
		baz := NewParam(mat.NewScalar(3.))

		m := &modelType{
			StrMap:  new(sync.Map),
			IntMap:  new(sync.Map),
			UintMap: new(sync.Map),
		}

		m.StrMap.Store("Foo", foo)
		m.IntMap.Store(42, bar)
		m.UintMap.Store(uint(3), baz)

		params := []collectedParam{
			{foo, "StrMap.Foo", Undefined},
			{bar, "IntMap.42", Undefined},
		}
		return traversalTest{
			name:                 name,
			model:                m,
			subMod:               true,
			expectedParams:       params,
			expectedParamsStrict: params,
			expectedModels:       []collectedModel{{m, ""}},
		}
	}(),
}

func TestForEachParam(t *testing.T) {
	for _, tt := range traversalTests {
		t.Run(tt.name, func(t *testing.T) {
			var actual []collectedParam
			ForEachParam(tt.model, func(p Param, n string, pt ParamsType) {
				actual = append(actual, collectedParam{param: p, name: n, pType: pt})
			})
			assert.Equal(t, tt.expectedParams, actual)
		})
	}
}

func TestForEachParamStrict(t *testing.T) {
	for _, tt := range traversalTests {
		t.Run(tt.name, func(t *testing.T) {
			var actual []collectedParam
			ForEachParamStrict(tt.model, func(p Param, n string, pt ParamsType) {
				actual = append(actual, collectedParam{param: p, name: n, pType: pt})
			})
			assert.Equal(t, tt.expectedParamsStrict, actual)
		})
	}
}

func TestForEachModel(t *testing.T) {
	for _, tt := range traversalTests {
		t.Run(tt.name, func(t *testing.T) {
			var actual []collectedModel
			ForEachModel(tt.model, func(m Model, n string) {
				actual = append(actual, collectedModel{model: m, name: n})
			})
			assert.Equal(t, tt.expectedModels, actual)
		})
	}
}
