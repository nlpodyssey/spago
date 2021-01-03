// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestBinaryMarshaling(t *testing.T) {
	var buf []byte

	{
		m := NewDense(2, 3, []Float{
			1, 2, 3,
			4, 5, 6,
		})
		w := bytes.NewBuffer(make([]byte, 0))
		n, err := MarshalBinaryTo(m, w)
		assert.Nil(t, err)
		assert.Greater(t, n, 0)
		buf = w.Bytes()
	}
	{
		m := NewEmptyDense(2, 3)
		n, err := UnmarshalBinaryFrom(m, bytes.NewBuffer(buf))
		assert.Nil(t, err)
		assert.Greater(t, n, 0)
		assert.Equal(t, []Float{1, 2, 3, 4, 5, 6}, m.Data())
	}
	{
		m, n, err := NewUnmarshalBinaryFrom(bytes.NewBuffer(buf))
		assert.Nil(t, err)
		assert.Greater(t, n, 0)
		assert.Equal(t, 2, m.Rows())
		assert.Equal(t, 3, m.Columns())
		assert.Equal(t, []Float{1, 2, 3, 4, 5, 6}, m.Data())
	}
}

func TestBinarySliceMarshaling(t *testing.T) {
	var buf []byte

	{
		ms := []Matrix{
			NewDense(2, 3, []Float{
				1, 2, 3,
				4, 5, 6,
			}),
			NewDense(2, 3, []Float{
				9, 8, 7,
				6, 5, 4,
			}),
		}
		w := bytes.NewBuffer(make([]byte, 0))
		n, err := MarshalBinarySlice(ms, w)
		assert.Nil(t, err)
		assert.Greater(t, n, 0)
		buf = w.Bytes()
	}
	{
		ms := []Matrix{
			NewEmptyDense(2, 3),
			NewEmptyDense(2, 3),
		}
		n, err := UnmarshalBinarySlice(ms, bytes.NewBuffer(buf))
		assert.Nil(t, err)
		assert.Greater(t, n, 0)
		assert.Equal(t, []Float{1, 2, 3, 4, 5, 6}, ms[0].Data())
		assert.Equal(t, []Float{9, 8, 7, 6, 5, 4}, ms[1].Data())
	}
	{
		ms := make([]Matrix, 2)
		n, err := NewUnmarshalBinarySlice(ms, bytes.NewBuffer(buf))
		assert.Nil(t, err)
		assert.Greater(t, n, 0)

		assert.Equal(t, 2, ms[0].Rows())
		assert.Equal(t, 3, ms[0].Columns())
		assert.Equal(t, []Float{1, 2, 3, 4, 5, 6}, ms[0].Data())

		assert.Equal(t, 2, ms[1].Rows())
		assert.Equal(t, 3, ms[1].Columns())
		assert.Equal(t, []Float{9, 8, 7, 6, 5, 4}, ms[1].Data())
	}
}
