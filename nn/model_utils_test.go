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

	m2 := Init(m)
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
