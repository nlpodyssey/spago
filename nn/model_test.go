// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestIntrospect(t *testing.T) {
	type T = float32

	type OtherModel struct {
		Module
		Baz Param `spago:"type:weights"`
		Qux Param
	}

	type Model struct {
		Module
		Foo   Param `spago:"type:biases"`
		Bar   Param
		Other *OtherModel
	}

	m := &Model{
		Foo: NewParam(mat.NewScalar[T](1)),
		Bar: NewParam(mat.NewScalar[T](2)),
		Other: &OtherModel{
			Baz: NewParam(mat.NewScalar[T](3)),
			Qux: NewParam(mat.NewScalar[T](4)),
		},
	}

	assert.Equal(t, "", m.Foo.Name())
	assert.Equal(t, Undefined, m.Foo.Type())

	assert.Equal(t, "", m.Bar.Name())
	assert.Equal(t, Undefined, m.Bar.Type())

	assert.Equal(t, "", m.Other.Baz.Name())
	assert.Equal(t, Undefined, m.Other.Baz.Type())

	assert.Equal(t, "", m.Other.Qux.Name())
	assert.Equal(t, Undefined, m.Other.Qux.Type())

	m2 := Introspect(m)
	assert.Same(t, m, m2)

	assert.Equal(t, "Foo", m.Foo.Name())
	assert.Equal(t, Biases, m.Foo.Type())

	assert.Equal(t, "Bar", m.Bar.Name())
	assert.Equal(t, Undefined, m.Bar.Type())

	assert.Equal(t, "Baz", m.Other.Baz.Name())
	assert.Equal(t, Weights, m.Other.Baz.Type())

	assert.Equal(t, "Qux", m.Other.Qux.Name())
	assert.Equal(t, Undefined, m.Other.Qux.Type())
}

func TestApply(t *testing.T) {
	for _, tt := range traversalTests {
		t.Run(tt.name, func(t *testing.T) {
			var actual []collectedModel
			Apply(tt.model, func(m Model, n string) {
				actual = append(actual, collectedModel{model: m, name: n})
			})
			assert.Equal(t, tt.expectedModels, actual)
		})
	}
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
