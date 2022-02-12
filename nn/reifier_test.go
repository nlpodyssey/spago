// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

var _ Model[float32] = &reifBaseModel[float32]{}

// reifBaseModel can be used as base Model in tests.
// The sole purpose of this struct is to satisfy the Model interface,
// providing a fake Forward method.
type reifBaseModel[T mat.DType] struct {
	BaseModel[T]
}

func (p reifBaseModel[_]) Forward(_ interface{}) interface{} {
	panic("this should never be called")
}

func TestModelContextualizer(t *testing.T) {
	t.Run("float32", testModelContextualizer[float32])
	t.Run("float64", testModelContextualizer[float64])
}

type reifModel1[T mat.DType] struct {
	reifBaseModel[T]
	ID int
}

type reifModel2[T mat.DType] struct {
	reifBaseModel[T]
	A Param[T]
}

type reifModel3[T mat.DType] struct {
	reifBaseModel[T]
	A Param[T]
	B Param[T]
}

type reifModel4[T mat.DType] struct {
	reifBaseModel[T]
	A []Param[T]
}

type reifStruct5[T mat.DType] struct {
	A Param[T]
	X int
	Z *reifStruct5[T] `spago:"type:params"`
}

type reifModel5[T mat.DType] struct {
	reifBaseModel[T]
	Foo reifStruct5[T]
	Bar reifStruct5[T] `spago:"type:params"`
}

type reifStruct6 struct {
	X int
}

type reifModel6[T mat.DType] struct {
	reifBaseModel[T]
	Foo reifStruct6
	Bar reifStruct6  `spago:"scope:processor"`
	Baz *reifStruct6 `spago:"scope:processor"`
}

type reifStruct7[T mat.DType] struct {
	P Param[T]
}

type reifModel7[T mat.DType] struct {
	reifBaseModel[T]
	Foo []reifStruct7[T]
	Bar []reifStruct7[T] `spago:"type:params"`
	Baz []*reifStruct7[T]
	Qux []*reifStruct7[T] `spago:"type:params"`
}

type reifModel8[T mat.DType] struct {
	reifBaseModel[T]
	Foo []int `spago:"type:params"`
}

type reifModel9[T mat.DType] struct {
	reifBaseModel[T]
	A map[string]Param[T]
}

type reifStruct10[T mat.DType] struct {
	P Param[T]
}

type reifModel10[T mat.DType] struct {
	reifBaseModel[T]
	Foo map[string]reifStruct10[T]
	Bar map[string]reifStruct10[T] `spago:"type:params"`
	Baz map[string]*reifStruct10[T]
	Qux map[string]*reifStruct10[T] `spago:"type:params"`
}

type reifModel11[T mat.DType] struct {
	reifBaseModel[T]
	Foo map[string]int `spago:"type:params"`
}

func testModelContextualizer[T mat.DType](t *testing.T) {
	t.Parallel()

	t.Run("model with irrelevant fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel1[T]{ID: 42}
		g := ag.NewGraph[T]()
		result := ReifyForTraining(sourceModel, g)
		assert.IsType(t, &reifModel1[T]{}, result)
		assert.NotSame(t, sourceModel, result)
		assert.Equal(t, &reifModel1[T]{ID: 42}, result)
	})

	t.Run("it panic if a model Param is not a *param", func(t *testing.T) {
		t.Parallel()

		g := ag.NewGraph[T]()
		sourceModel := &reifModel2[T]{
			A: NewParam[T](mat.NewScalar[T](1)).(*param[T]).wrappedParam(g),
		}

		assert.Panics(t, func() {
			Reify(sourceModel, g, Training)
		})
	})

	t.Run("it contextualizes Param fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel3[T]{
			A: NewParam[T](mat.NewScalar[T](1)),
			B: NewParam[T](mat.NewScalar[T](2)),
		}
		g := ag.NewGraph[T]()
		result := ReifyForTraining(sourceModel, g)

		assert.IsType(t, &wrappedParam[T]{}, result.A)
		assert.IsType(t, &wrappedParam[T]{}, result.B)
		assert.Same(t, sourceModel.A, result.A.(*wrappedParam[T]).param)
		assert.Same(t, sourceModel.B, result.B.(*wrappedParam[T]).param)
	})

	t.Run("it contextualizes []Param fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel4[T]{
			A: []Param[T]{
				NewParam[T](mat.NewScalar[T](1)),
				NewParam[T](mat.NewScalar[T](2)),
			},
		}
		g := ag.NewGraph[T]()
		result := ReifyForTraining(sourceModel, g)

		assert.IsType(t, &wrappedParam[T]{}, result.A[0])
		assert.IsType(t, &wrappedParam[T]{}, result.A[1])
		assert.Same(t, sourceModel.A[0], result.A[0].(*wrappedParam[T]).param)
		assert.Same(t, sourceModel.A[1], result.A[1].(*wrappedParam[T]).param)
	})

	t.Run("it contextualizes tagged nested struct fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel5[T]{
			Foo: reifStruct5[T]{
				A: NewParam[T](mat.NewScalar[T](1)),
				X: 11,
				Z: &reifStruct5[T]{
					A: NewParam[T](mat.NewScalar[T](2)),
					X: 22,
					Z: nil,
				},
			},
			Bar: reifStruct5[T]{
				A: NewParam[T](mat.NewScalar[T](10)),
				X: 33,
				Z: &reifStruct5[T]{
					A: NewParam[T](mat.NewScalar[T](20)),
					X: 44,
					Z: nil,
				},
			},
		}
		g := ag.NewGraph[T]()
		result := ReifyForTraining(sourceModel, g)

		assert.Equal(t, sourceModel.Foo, result.Foo)
		assert.Same(t, sourceModel.Foo.Z, result.Foo.Z)
		// Paranoid checks to be sure the source model was not illegally modified
		assert.IsType(t, &param[T]{}, result.Foo.A)
		assert.IsType(t, &param[T]{}, result.Foo.Z.A)

		assert.NotEqual(t, sourceModel.Bar, result.Bar)
		assert.NotSame(t, sourceModel.Bar.Z, result.Bar.Z)

		assert.IsType(t, &wrappedParam[T]{}, result.Bar.A)
		assert.IsType(t, &wrappedParam[T]{}, result.Bar.Z.A)
		assert.Same(t, sourceModel.Bar.A, result.Bar.A.(*wrappedParam[T]).param)
		assert.Same(t, sourceModel.Bar.Z.A, result.Bar.Z.A.(*wrappedParam[T]).param)

		// Be sure X's were copied
		assert.Equal(t, 11, result.Foo.X)
		assert.Equal(t, 22, result.Foo.Z.X)
		assert.Equal(t, 33, result.Bar.X)
		assert.Equal(t, 44, result.Bar.Z.X)
	})

	t.Run("it totally ignores fields tagged as 'processor'", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel6[T]{
			Foo: reifStruct6{X: 11},
			// It's unusual to set a value on a "processor"-scoped field, but
			// here it's useful to ensure that it is actually ignored
			Bar: reifStruct6{X: 22},
			Baz: &reifStruct6{X: 33},
		}
		g := ag.NewGraph[T]()
		result := ReifyForTraining(sourceModel, g)

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

		sourceModel := &reifModel7[T]{
			Foo: []reifStruct7[T]{{P: NewParam[T](mat.NewScalar[T](1))}},
			Bar: []reifStruct7[T]{{P: NewParam[T](mat.NewScalar[T](2))}},
			Baz: []*reifStruct7[T]{{P: NewParam[T](mat.NewScalar[T](3))}},
			Qux: []*reifStruct7[T]{{P: NewParam[T](mat.NewScalar[T](4))}},
		}
		g := ag.NewGraph[T]()
		result := ReifyForTraining(sourceModel, g)

		assert.Equal(t, sourceModel.Foo, result.Foo)
		assert.Same(t, sourceModel.Baz[0], result.Baz[0])

		assert.NotEqual(t, sourceModel.Bar, result.Foo)
		assert.NotSame(t, sourceModel.Qux[0], result.Baz[0])

		assert.IsType(t, &wrappedParam[T]{}, result.Bar[0].P)
		assert.IsType(t, &wrappedParam[T]{}, result.Qux[0].P)

		// Paranoid checks to be sure the source model was not illegally modified
		assert.IsType(t, &param[T]{}, sourceModel.Foo[0].P)
		assert.IsType(t, &param[T]{}, sourceModel.Bar[0].P)
		assert.IsType(t, &param[T]{}, sourceModel.Baz[0].P)
		assert.IsType(t, &param[T]{}, sourceModel.Qux[0].P)
	})

	t.Run("it panics with tagged slices of elements which are not structs nor pointers", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel8[T]{
			Foo: []int{1, 2, 3},
		}
		g := ag.NewGraph[T]()

		assert.Panics(t, func() {
			Reify(sourceModel, g, Training)
		})
	})

	t.Run("it contextualizes map[...]Param fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel9[T]{
			A: map[string]Param[T]{
				"a": NewParam[T](mat.NewScalar[T](1)),
				"b": NewParam[T](mat.NewScalar[T](2)),
			},
		}
		g := ag.NewGraph[T]()
		result := ReifyForTraining(sourceModel, g)

		assert.IsType(t, &wrappedParam[T]{}, result.A["a"])
		assert.IsType(t, &wrappedParam[T]{}, result.A["b"])
		assert.Same(t, sourceModel.A["a"], result.A["a"].(*wrappedParam[T]).param)
		assert.Same(t, sourceModel.A["b"], result.A["b"].(*wrappedParam[T]).param)
	})

	t.Run("it contextualizes tagged maps of structs or pointers", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel10[T]{
			Foo: map[string]reifStruct10[T]{"a": {P: NewParam[T](mat.NewScalar[T](1))}},
			Bar: map[string]reifStruct10[T]{"b": {P: NewParam[T](mat.NewScalar[T](2))}},
			Baz: map[string]*reifStruct10[T]{"c": {P: NewParam[T](mat.NewScalar[T](3))}},
			Qux: map[string]*reifStruct10[T]{"d": {P: NewParam[T](mat.NewScalar[T](4))}},
		}
		g := ag.NewGraph[T]()
		result := ReifyForTraining(sourceModel, g)

		assert.Equal(t, sourceModel.Foo, result.Foo)
		assert.Same(t, sourceModel.Baz["c"], result.Baz["c"])

		assert.NotEqual(t, sourceModel.Bar, result.Foo)
		assert.NotSame(t, sourceModel.Qux["d"], result.Baz["c"])

		assert.IsType(t, &wrappedParam[T]{}, result.Bar["b"].P)
		assert.IsType(t, &wrappedParam[T]{}, result.Qux["d"].P)

		// Paranoid checks to be sure the source model was not illegally modified
		assert.IsType(t, &param[T]{}, sourceModel.Foo["a"].P)
		assert.IsType(t, &param[T]{}, sourceModel.Bar["b"].P)
		assert.IsType(t, &param[T]{}, sourceModel.Baz["c"].P)
		assert.IsType(t, &param[T]{}, sourceModel.Qux["d"].P)
	})

	t.Run("it panics with tagged maps of elements which are not structs nor pointers", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel11[T]{
			Foo: map[string]int{"a": 1, "b": 2},
		}
		g := ag.NewGraph[T]()

		assert.Panics(t, func() {
			Reify(sourceModel, g, Training)
		})
	})
}
