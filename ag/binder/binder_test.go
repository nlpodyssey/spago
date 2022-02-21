// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package binder

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

var _ nn.Model[float32] = &reifBaseModel[float32]{}

// reifBaseModel can be used as base Model in tests.
// The sole purpose of this struct is to satisfy the Model interface,
// providing a fake Forward method.
type reifBaseModel[T mat.DType] struct {
	nn.BaseModel[T]
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
	A nn.Param[T]
}

type reifModel3[T mat.DType] struct {
	reifBaseModel[T]
	A nn.Param[T]
	B nn.Param[T]
}

type reifModel4[T mat.DType] struct {
	reifBaseModel[T]
	A []nn.Param[T]
}

type reifStruct5[T mat.DType] struct {
	ag.Differentiable[T]
	A nn.Param[T]
	X int
	Z *reifStruct5[T]
}

type reifModel5[T mat.DType] struct {
	reifBaseModel[T]
	Foo reifStruct5[T]
	Bar reifStruct5[T]
}

type reifStruct7[T mat.DType] struct {
	ag.Differentiable[T]
	P nn.Param[T]
}

type reifModel7[T mat.DType] struct {
	reifBaseModel[T]
	Foo []reifStruct7[T]
	Bar []reifStruct7[T]
	Baz []*reifStruct7[T]
	Qux []*reifStruct7[T]
}

type reifModel8[T mat.DType] struct {
	reifBaseModel[T]
	Foo []int
}

type reifModel9[T mat.DType] struct {
	reifBaseModel[T]
	A map[string]nn.Param[T]
}

func testModelContextualizer[T mat.DType](t *testing.T) {
	t.Parallel()

	t.Run("model with irrelevant fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel1[T]{ID: 42}
		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		result := Bind(g, sourceModel)
		assert.IsType(t, &reifModel1[T]{}, result)
		assert.NotSame(t, sourceModel, result)
		assert.Equal(t, &reifModel1[T]{ID: 42}, result)
	})

	t.Run("it panic if a model Param is not a *BaseParam", func(t *testing.T) {
		t.Parallel()

		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		p := nn.NewParam[T](mat.NewScalar[T](1)).(*nn.BaseParam[T])
		sourceModel := &reifModel2[T]{
			A: &nn.ParamNode[T]{Param: p, Node: g.NewWrap(p)},
		}

		assert.Panics(t, func() {
			Bind(g, sourceModel)
		})
	})

	t.Run("it contextualizes Param fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel3[T]{
			A: nn.NewParam[T](mat.NewScalar[T](1)),
			B: nn.NewParam[T](mat.NewScalar[T](2)),
		}
		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		result := Bind(g, sourceModel)

		assert.IsType(t, &nn.ParamNode[T]{}, result.A)
		assert.IsType(t, &nn.ParamNode[T]{}, result.B)
		assert.Same(t, sourceModel.A, result.A.(*nn.ParamNode[T]).Param)
		assert.Same(t, sourceModel.B, result.B.(*nn.ParamNode[T]).Param)
	})

	t.Run("it contextualizes []Param fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel4[T]{
			A: []nn.Param[T]{
				nn.NewParam[T](mat.NewScalar[T](1)),
				nn.NewParam[T](mat.NewScalar[T](2)),
			},
		}
		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		result := Bind(g, sourceModel)

		assert.IsType(t, &nn.ParamNode[T]{}, result.A[0])
		assert.IsType(t, &nn.ParamNode[T]{}, result.A[1])
		assert.Same(t, sourceModel.A[0], result.A[0].(*nn.ParamNode[T]).Param)
		assert.Same(t, sourceModel.A[1], result.A[1].(*nn.ParamNode[T]).Param)
	})

	t.Run("it panics with a model with an already reified param", func(t *testing.T) {
		t.Parallel()

		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		p := nn.NewParam[T](mat.NewScalar[T](1))

		sourceModel := &reifModel2[T]{
			A: &nn.ParamNode[T]{Param: p, Node: g.NewWrap(p)},
		}

		assert.Panics(t, func() {
			Bind(g, sourceModel)
		})
	})

	t.Run("it contextualizes map[...]Param fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel9[T]{
			A: map[string]nn.Param[T]{
				"a": nn.NewParam[T](mat.NewScalar[T](1)),
				"b": nn.NewParam[T](mat.NewScalar[T](2)),
			},
		}
		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		result := Bind(g, sourceModel)

		assert.IsType(t, &nn.ParamNode[T]{}, result.A["a"])
		assert.IsType(t, &nn.ParamNode[T]{}, result.A["b"])
		assert.Same(t, sourceModel.A["a"], result.A["a"].(*nn.ParamNode[T]).Param)
		assert.Same(t, sourceModel.A["b"], result.A["b"].(*nn.ParamNode[T]).Param)
	})
}
