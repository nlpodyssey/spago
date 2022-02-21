// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag_test

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModelContextualizer(t *testing.T) {
	t.Run("float32", testModelContextualizer[float32])
	t.Run("float64", testModelContextualizer[float64])
}

type paramInterface[T mat.DType] interface {
	ag.Node[T]
}

type paramStruct[T mat.DType] struct {
	m mat.Matrix[T]
}

func (ps *paramStruct[T]) Value() mat.Matrix[T]          { panic("should not be called") }
func (ps *paramStruct[T]) ScalarValue() T                { panic("should not be called") }
func (ps *paramStruct[T]) Grad() mat.Matrix[T]           { panic("should not be called") }
func (ps *paramStruct[_]) HasGrad() bool                 { panic("should not be called") }
func (ps *paramStruct[_]) RequiresGrad() bool            { panic("should not be called") }
func (ps *paramStruct[T]) PropagateGrad(_ mat.Matrix[T]) { panic("should not be called") }
func (ps *paramStruct[_]) ZeroGrad()                     { panic("should not be called") }
func (ps *paramStruct[T]) Graph() *ag.Graph[T]           { panic("should not be called") }
func (ps *paramStruct[_]) ID() int                       { panic("should not be called") }
func (ps *paramStruct[_]) TimeStep() int                 { panic("should not be called") }

var _ paramInterface[float32] = &paramStruct[float32]{}

func (ps *paramStruct[T]) Bind(g *ag.Graph[T]) ag.Node[T] {
	return &paramNodeStruct[T]{
		paramInterface: ps,
		g:              g,
	}
}

type paramNodeStruct[T mat.DType] struct {
	paramInterface[T]
	g *ag.Graph[T]
}

type reifModel1[T mat.DType] struct {
	ag.DifferentiableModule[T]
	ID int
}

type reifModel2[T mat.DType] struct {
	ag.DifferentiableModule[T]
	A paramInterface[T]
}

type reifModel3[T mat.DType] struct {
	ag.DifferentiableModule[T]
	A paramInterface[T]
	B paramInterface[T]
}

type reifModel4[T mat.DType] struct {
	ag.DifferentiableModule[T]
	A []paramInterface[T]
}

type reifStruct5[T mat.DType] struct {
	ag.Differentiable[T]
	A paramInterface[T]
	X int
	Z *reifStruct5[T]
}

type reifModel5[T mat.DType] struct {
	ag.DifferentiableModule[T]
	Foo reifStruct5[T]
	Bar reifStruct5[T]
}

type reifStruct7[T mat.DType] struct {
	ag.Differentiable[T]
	P paramInterface[T]
}

type reifModel7[T mat.DType] struct {
	ag.DifferentiableModule[T]
	Foo []reifStruct7[T]
	Bar []reifStruct7[T]
	Baz []*reifStruct7[T]
	Qux []*reifStruct7[T]
}

type reifModel8[T mat.DType] struct {
	ag.DifferentiableModule[T]
	Foo []int
}

type reifModel9[T mat.DType] struct {
	ag.DifferentiableModule[T]
	A map[string]paramInterface[T]
}

func testModelContextualizer[T mat.DType](t *testing.T) {
	t.Parallel()

	t.Run("model with irrelevant fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel1[T]{ID: 42}
		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		result := ag.Bind(g, sourceModel)
		assert.IsType(t, &reifModel1[T]{}, result)
		assert.NotSame(t, sourceModel, result)
		assert.Equal(t, &reifModel1[T]{
			DifferentiableModule: ag.DifferentiableModule[T]{Graph: g},
			ID:                   42,
		}, result)
	})

	t.Run("it contextualizes Param fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel3[T]{
			A: &paramStruct[T]{mat.NewScalar[T](1)},
			B: &paramStruct[T]{mat.NewScalar[T](2)},
		}
		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		result := ag.Bind(g, sourceModel)

		_ = result
		assert.IsType(t, &paramNodeStruct[T]{}, result.A)
		assert.IsType(t, &paramNodeStruct[T]{}, result.B)
		assert.Same(t, sourceModel.A, result.A.(*paramNodeStruct[T]).paramInterface)
		assert.Same(t, sourceModel.B, result.B.(*paramNodeStruct[T]).paramInterface)
	})

	t.Run("it contextualizes []Param fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel4[T]{
			A: []paramInterface[T]{
				&paramStruct[T]{mat.NewScalar[T](1)},
				&paramStruct[T]{mat.NewScalar[T](2)},
			},
		}
		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		result := ag.Bind(g, sourceModel)

		_ = result
		assert.IsType(t, &paramNodeStruct[T]{}, result.A[0])
		assert.IsType(t, &paramNodeStruct[T]{}, result.A[1])
		assert.Same(t, sourceModel.A[0], result.A[0].(*paramNodeStruct[T]).paramInterface)
		assert.Same(t, sourceModel.A[1], result.A[1].(*paramNodeStruct[T]).paramInterface)
	})

	t.Run("it panics with a model with an already reified param", func(t *testing.T) {
		t.Parallel()

		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		p := &paramStruct[T]{mat.NewScalar[T](1)}

		sourceModel := &reifModel2[T]{
			A: &paramNodeStruct[T]{paramInterface: p, g: g},
		}

		assert.Panics(t, func() {
			ag.Bind(g, sourceModel)
		})
	})

	t.Run("it contextualizes map[...]Param fields", func(t *testing.T) {
		t.Parallel()

		sourceModel := &reifModel9[T]{
			A: map[string]paramInterface[T]{
				"a": &paramStruct[T]{mat.NewScalar[T](1)},
				"b": &paramStruct[T]{mat.NewScalar[T](2)},
			},
		}
		g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
		result := ag.Bind(g, sourceModel)

		_ = result
		assert.IsType(t, &paramNodeStruct[T]{}, result.A["a"])
		assert.IsType(t, &paramNodeStruct[T]{}, result.A["b"])
		assert.Same(t, sourceModel.A["a"], result.A["a"].(*paramNodeStruct[T]).paramInterface)
		assert.Same(t, sourceModel.A["b"], result.A["b"].(*paramNodeStruct[T]).paramInterface)
	})
}
