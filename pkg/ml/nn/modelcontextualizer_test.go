// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/stretchr/testify/assert"
	"testing"
)

// ModelContextualizerBaseModel can be used as base Model in tests.
// The sole purpose of this struct is to satisfy the Model interface,
// providing a fake NewProc method.
type ModelContextualizerBaseModel struct{}

var _ Model = &ModelContextualizerBaseModel{}

func (p ModelContextualizerBaseModel) NewProc(_ Context) Processor {
	panic("this should never be called")
}

func TestModelContextualizer(t *testing.T) {
	t.Parallel()

	t.Run("model with irrelevant fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ModelContextualizerBaseModel
			ID int
		}

		sourceModel := &TestModel{ID: 42}
		g := ag.NewGraph()
		result := newModelContextualizer(g).contextualize(sourceModel)
		assert.IsType(t, &TestModel{}, result)
		assert.NotSame(t, sourceModel, result)
		assert.Equal(t, &TestModel{ID: 42}, result)
	})

	t.Run("it panic if a model Param is not a *param", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ModelContextualizerBaseModel
			A Param
		}

		g := ag.NewGraph()
		sourceModel := &TestModel{
			A: NewParam(mat.NewScalar(1)).(*param).wrappedParam(g),
		}
		mc := newModelContextualizer(g)

		assert.Panics(t, func() {
			mc.contextualize(sourceModel)
		})
	})

	t.Run("it contextualizes Param fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ModelContextualizerBaseModel
			A Param
			B Param
		}

		sourceModel := &TestModel{
			A: NewParam(mat.NewScalar(1)),
			B: NewParam(mat.NewScalar(2)),
		}
		g := ag.NewGraph()
		result := newModelContextualizer(g).contextualize(sourceModel).(*TestModel)

		assert.IsType(t, &wrappedParam{}, result.A)
		assert.IsType(t, &wrappedParam{}, result.B)
		assert.Same(t, sourceModel.A, result.A.(*wrappedParam).Param)
		assert.Same(t, sourceModel.B, result.B.(*wrappedParam).Param)
	})

	t.Run("it contextualizes []Param fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ModelContextualizerBaseModel
			A []Param
		}

		sourceModel := &TestModel{
			A: []Param{
				NewParam(mat.NewScalar(1)),
				NewParam(mat.NewScalar(2)),
			},
		}
		g := ag.NewGraph()
		result := newModelContextualizer(g).contextualize(sourceModel).(*TestModel)

		assert.IsType(t, &wrappedParam{}, result.A[0])
		assert.IsType(t, &wrappedParam{}, result.A[1])
		assert.Same(t, sourceModel.A[0], result.A[0].(*wrappedParam).Param)
		assert.Same(t, sourceModel.A[1], result.A[1].(*wrappedParam).Param)
	})

	t.Run("it contextualizes tagged nested struct fields", func(t *testing.T) {
		t.Parallel()

		type MyStruct struct {
			A Param
			X int
			Z *MyStruct `type:"params"`
		}

		type TestModel struct {
			ModelContextualizerBaseModel
			Foo MyStruct
			Bar MyStruct `type:"params"`
		}

		sourceModel := &TestModel{
			Foo: MyStruct{
				A: NewParam(mat.NewScalar(1)),
				X: 11,
				Z: &MyStruct{
					A: NewParam(mat.NewScalar(2)),
					X: 22,
					Z: nil,
				},
			},
			Bar: MyStruct{
				A: NewParam(mat.NewScalar(10)),
				X: 33,
				Z: &MyStruct{
					A: NewParam(mat.NewScalar(20)),
					X: 44,
					Z: nil,
				},
			},
		}
		g := ag.NewGraph()
		result := newModelContextualizer(g).contextualize(sourceModel).(*TestModel)

		assert.Equal(t, sourceModel.Foo, result.Foo)
		assert.Same(t, sourceModel.Foo.Z, result.Foo.Z)
		// Paranoid checks to be sure the source model was not illegally modified
		assert.IsType(t, &param{}, result.Foo.A)
		assert.IsType(t, &param{}, result.Foo.Z.A)

		assert.NotEqual(t, sourceModel.Bar, result.Bar)
		assert.NotSame(t, sourceModel.Bar.Z, result.Bar.Z)

		assert.IsType(t, &wrappedParam{}, result.Bar.A)
		assert.IsType(t, &wrappedParam{}, result.Bar.Z.A)
		assert.Same(t, sourceModel.Bar.A, result.Bar.A.(*wrappedParam).Param)
		assert.Same(t, sourceModel.Bar.Z.A, result.Bar.Z.A.(*wrappedParam).Param)

		// Be sure X's were copied
		assert.Equal(t, 11, result.Foo.X)
		assert.Equal(t, 22, result.Foo.Z.X)
		assert.Equal(t, 33, result.Bar.X)
		assert.Equal(t, 44, result.Bar.Z.X)
	})

	t.Run("it totally ignores fields tagged as 'processor'", func(t *testing.T) {
		t.Parallel()

		type MyStruct struct {
			X int
		}

		type TestModel struct {
			ModelContextualizerBaseModel
			Foo MyStruct
			Bar MyStruct  `scope:"processor"`
			Baz *MyStruct `scope:"processor"`
		}

		sourceModel := &TestModel{
			Foo: MyStruct{X: 11},
			// It's unusual to set a value on a "processor"-scoped field, but
			// here it's useful to ensure that it is actually ignored
			Bar: MyStruct{X: 22},
			Baz: &MyStruct{X: 33},
		}
		g := ag.NewGraph()
		result := newModelContextualizer(g).contextualize(sourceModel).(*TestModel)

		assert.Equal(t, 11, result.Foo.X)
		assert.Equal(t, 0, result.Bar.X)
		assert.Nil(t, result.Baz)

		// Paranoid check to be sure that the source model was not modified
		assert.Equal(t, 11, sourceModel.Foo.X)
		assert.Equal(t, 22, sourceModel.Bar.X)
		assert.NotNil(t, sourceModel.Baz)
		assert.Equal(t, 33, sourceModel.Baz.X)
	})

	t.Run("it contextualizes tagged slices of structs or pointers", func(t *testing.T) {
		t.Parallel()

		type MyStruct struct {
			P Param
		}

		type TestModel struct {
			ModelContextualizerBaseModel
			Foo []MyStruct
			Bar []MyStruct `type:"params"`
			Baz []*MyStruct
			Qux []*MyStruct `type:"params"`
		}

		sourceModel := &TestModel{
			Foo: []MyStruct{{P: NewParam(mat.NewScalar(1))}},
			Bar: []MyStruct{{P: NewParam(mat.NewScalar(2))}},
			Baz: []*MyStruct{{P: NewParam(mat.NewScalar(3))}},
			Qux: []*MyStruct{{P: NewParam(mat.NewScalar(4))}},
		}
		g := ag.NewGraph()
		result := newModelContextualizer(g).contextualize(sourceModel).(*TestModel)

		assert.Equal(t, sourceModel.Foo, result.Foo)
		assert.Same(t, sourceModel.Baz[0], result.Baz[0])

		assert.NotEqual(t, sourceModel.Bar, result.Foo)
		assert.NotSame(t, sourceModel.Qux[0], result.Baz[0])

		assert.IsType(t, &wrappedParam{}, result.Bar[0].P)
		assert.IsType(t, &wrappedParam{}, result.Qux[0].P)

		// Paranoid checks to be sure the source model was not illegally modified
		assert.IsType(t, &param{}, sourceModel.Foo[0].P)
		assert.IsType(t, &param{}, sourceModel.Bar[0].P)
		assert.IsType(t, &param{}, sourceModel.Baz[0].P)
		assert.IsType(t, &param{}, sourceModel.Qux[0].P)
	})

	t.Run("it panics with tagged slices of elements which are not structs nor pointers", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ModelContextualizerBaseModel
			Foo []int `type:"params"`
		}

		sourceModel := &TestModel{
			Foo: []int{1, 2, 3},
		}
		g := ag.NewGraph()
		mc := newModelContextualizer(g)

		assert.Panics(t, func() {
			mc.contextualize(sourceModel)
		})
	})

	t.Run("it contextualizes map[...]Param fields", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ModelContextualizerBaseModel
			A map[string]Param
		}

		sourceModel := &TestModel{
			A: map[string]Param{
				"a": NewParam(mat.NewScalar(1)),
				"b": NewParam(mat.NewScalar(2)),
			},
		}
		g := ag.NewGraph()
		result := newModelContextualizer(g).contextualize(sourceModel).(*TestModel)

		assert.IsType(t, &wrappedParam{}, result.A["a"])
		assert.IsType(t, &wrappedParam{}, result.A["b"])
		assert.Same(t, sourceModel.A["a"], result.A["a"].(*wrappedParam).Param)
		assert.Same(t, sourceModel.A["b"], result.A["b"].(*wrappedParam).Param)
	})

	t.Run("it contextualizes tagged maps of structs or pointers", func(t *testing.T) {
		t.Parallel()

		type MyStruct struct {
			P Param
		}

		type TestModel struct {
			ModelContextualizerBaseModel
			Foo map[string]MyStruct
			Bar map[string]MyStruct `type:"params"`
			Baz map[string]*MyStruct
			Qux map[string]*MyStruct `type:"params"`
		}

		sourceModel := &TestModel{
			Foo: map[string]MyStruct{"a": {P: NewParam(mat.NewScalar(1))}},
			Bar: map[string]MyStruct{"b": {P: NewParam(mat.NewScalar(2))}},
			Baz: map[string]*MyStruct{"c": {P: NewParam(mat.NewScalar(3))}},
			Qux: map[string]*MyStruct{"d": {P: NewParam(mat.NewScalar(4))}},
		}
		g := ag.NewGraph()
		result := newModelContextualizer(g).contextualize(sourceModel).(*TestModel)

		assert.Equal(t, sourceModel.Foo, result.Foo)
		assert.Same(t, sourceModel.Baz["c"], result.Baz["c"])

		assert.NotEqual(t, sourceModel.Bar, result.Foo)
		assert.NotSame(t, sourceModel.Qux["d"], result.Baz["c"])

		assert.IsType(t, &wrappedParam{}, result.Bar["b"].P)
		assert.IsType(t, &wrappedParam{}, result.Qux["d"].P)

		// Paranoid checks to be sure the source model was not illegally modified
		assert.IsType(t, &param{}, sourceModel.Foo["a"].P)
		assert.IsType(t, &param{}, sourceModel.Bar["b"].P)
		assert.IsType(t, &param{}, sourceModel.Baz["c"].P)
		assert.IsType(t, &param{}, sourceModel.Qux["d"].P)
	})

	t.Run("it panics with tagged maps of elements which are not structs nor pointers", func(t *testing.T) {
		t.Parallel()

		type TestModel struct {
			ModelContextualizerBaseModel
			Foo map[string]int `type:"params"`
		}

		sourceModel := &TestModel{
			Foo: map[string]int{"a": 1, "b": 2},
		}
		g := ag.NewGraph()
		mc := newModelContextualizer(g)

		assert.Panics(t, func() {
			mc.contextualize(sourceModel)
		})
	})
}
