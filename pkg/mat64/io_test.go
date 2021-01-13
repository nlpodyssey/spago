// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

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
		err := MarshalBinaryTo(m, w)
		assert.Nil(t, err)
		buf = w.Bytes()
	}
	{
		m := NewEmptyDense(2, 3)
		err := UnmarshalBinaryFrom(m, bytes.NewBuffer(buf))
		assert.Nil(t, err)
		assert.Equal(t, []Float{1, 2, 3, 4, 5, 6}, m.Data())
	}
	{
		m, err := NewUnmarshalBinaryFrom(bytes.NewBuffer(buf))
		assert.Nil(t, err)
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
		err := MarshalBinarySlice(ms, w)
		assert.Nil(t, err)
		buf = w.Bytes()
	}
	{
		ms := []Matrix{
			NewEmptyDense(2, 3),
			NewEmptyDense(2, 3),
		}
		err := UnmarshalBinarySlice(ms, bytes.NewBuffer(buf))
		assert.Nil(t, err)
		assert.Equal(t, []Float{1, 2, 3, 4, 5, 6}, ms[0].Data())
		assert.Equal(t, []Float{9, 8, 7, 6, 5, 4}, ms[1].Data())
	}
	{
		ms := make([]Matrix, 2)
		err := NewUnmarshalBinarySlice(ms, bytes.NewBuffer(buf))
		assert.Nil(t, err)

		assert.Equal(t, 2, ms[0].Rows())
		assert.Equal(t, 3, ms[0].Columns())
		assert.Equal(t, []Float{1, 2, 3, 4, 5, 6}, ms[0].Data())

		assert.Equal(t, 2, ms[1].Rows())
		assert.Equal(t, 3, ms[1].Columns())
		assert.Equal(t, []Float{9, 8, 7, 6, 5, 4}, ms[1].Data())
	}
}
