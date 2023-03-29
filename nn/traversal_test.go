// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"sync"

	"github.com/nlpodyssey/spago/mat"
)

type collectedParam struct {
	param Param
}

type collectedModel struct {
	model Model
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
	fn func(callback func(param Param))
}

var _ ParamsTraverser = traversableType{}

func (t traversableType) TraverseParams(f func(param Param)) {
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
			expectedModels:       []collectedModel{{m}},
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
			expectedModels:       []collectedModel{{m}},
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
			{m.A},
			{m.B},
		}
		return traversalTest{
			name:                 name,
			model:                m,
			subMod:               false,
			expectedParams:       params,
			expectedParamsStrict: params,
			expectedModels:       []collectedModel{{m}},
		}
	}(),
	func() traversalTest {
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
			{m.A[0]},
			{m.A[1]},
			{m.B[0]},
			{m.B[1]},
		}
		return traversalTest{
			model:                m,
			subMod:               false,
			expectedParams:       params,
			expectedParamsStrict: params,
			expectedModels:       []collectedModel{{m}},
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
				{m.P},
				{nested.P},
			},
			expectedParamsStrict: []collectedParam{
				{m.P},
			},
			expectedModels: []collectedModel{
				{m},
				{nested},
				{nested.M},
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
				{m.P},
				{mA.P},
				{mB.P},
			},
			expectedParamsStrict: []collectedParam{
				{m.P},
			},
			expectedModels: []collectedModel{
				{m},
				{mA},
				{mB},
			},
		}
	}(),
	func() traversalTest {
		name := "nested model list type fields"

		type modelList []Model
		type modelType struct {
			Module
			P Param
			M modelList
		}

		mA := &modelType{P: NewParam(mat.NewScalar(100.))}
		mB := &modelType{P: NewParam(mat.NewScalar(200.))}
		m := &modelType{
			P: NewParam(mat.NewScalar(1.)),
			M: modelList{mA, mB},
		}

		return traversalTest{
			name:   name,
			model:  m,
			subMod: true,
			expectedParams: []collectedParam{
				{m.P},
				{mA.P},
				{mB.P},
			},
			expectedParamsStrict: []collectedParam{
				{m.P},
			},
			expectedModels: []collectedModel{
				{m},
				{mA},
				{mB},
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
				{m.P},
				{mA.P},
				{mB.P},
			},
			expectedParamsStrict: []collectedParam{
				{m.P},
			},
			expectedModels: []collectedModel{
				{m},
				{mA},
				{mB},
			},
		}
	}(),
	func() traversalTest {
		name := "nested Model items in array fields"

		type modelType struct {
			Module
			P Param
			M [2]any
		}

		mA := &modelType{P: NewParam(mat.NewScalar(100.))}
		mB := &modelType{P: NewParam(mat.NewScalar(200.))}
		m := &modelType{
			P: NewParam(mat.NewScalar(1.)),
			M: [2]any{mA, mB},
		}

		return traversalTest{
			name:   name,
			model:  m,
			subMod: true,
			expectedParams: []collectedParam{
				{m.P},
				{mA.P},
				{mB.P},
			},
			expectedParamsStrict: []collectedParam{
				{m.P},
			},
			expectedModels: []collectedModel{
				{m},
				{mA},
				{mB},
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
				{func(f func(param Param)) {
					f(delta)
					f(echo)
				}},
			},
		}
		m := &modelType{
			Foo: NewParam(mat.NewScalar(1.)),
			Bar: []simpleStruct{{P: NewParam(mat.NewScalar(2.))}},
			Baz: nested,
			Qux: []traversableType{
				{func(f func(param Param)) {
					f(alfa)
					f(bravo)
				}},
				{func(f func(param Param)) {
					f(charlie)
				}},
			},
		}

		return traversalTest{
			name:   name,
			model:  m,
			subMod: true,
			expectedParams: []collectedParam{
				{m.Foo},
				{nested.Foo},
				{delta},
				{echo},
				{alfa},
				{bravo},
				{charlie},
			},
			expectedParamsStrict: []collectedParam{
				{m.Foo},
				{alfa},
				{bravo},
				{charlie},
			},
			expectedModels: []collectedModel{
				{m},
				{nested},
			},
		}
	}(),
	func() traversalTest {
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
			{m.MI[0]},
			{m.MS["a"]},
		}
		return traversalTest{
			model:                m,
			subMod:               true,
			expectedParams:       params,
			expectedParamsStrict: params,
			expectedModels:       []collectedModel{{m}},
		}
	}(),
	func() traversalTest {
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
			{foo},
			{bar},
		}
		return traversalTest{
			model:                m,
			subMod:               true,
			expectedParams:       params,
			expectedParamsStrict: params,
			expectedModels:       []collectedModel{{m}},
		}
	}(),
}
