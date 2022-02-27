// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestIntrospect(t *testing.T) {
	type T = float32

	type OtherModel struct {
		BaseModel[T]
		Baz Param[T] `spago:"type:weights"`
		Qux Param[T]
	}

	type Model struct {
		BaseModel[T]
		Foo   Param[T] `spago:"type:biases"`
		Bar   Param[T]
		Other *OtherModel
	}

	m := &Model{
		Foo: NewParam[T](mat.NewScalar[T](1)),
		Bar: NewParam[T](mat.NewScalar[T](2)),
		Other: &OtherModel{
			Baz: NewParam[T](mat.NewScalar[T](3)),
			Qux: NewParam[T](mat.NewScalar[T](4)),
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

	m2 := Introspect[T](m)
	assert.Same(t, m, m2)

	assert.Equal(t, "foo", m.Foo.Name())
	assert.Equal(t, Biases, m.Foo.Type())

	assert.Equal(t, "bar", m.Bar.Name())
	assert.Equal(t, Undefined, m.Bar.Type())

	assert.Equal(t, "baz", m.Other.Baz.Name())
	assert.Equal(t, Weights, m.Other.Baz.Type())

	assert.Equal(t, "qux", m.Other.Qux.Name())
	assert.Equal(t, Undefined, m.Other.Qux.Type())
}
