// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"bytes"
	"encoding/gob"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDense_Gob(t *testing.T) {
	matrixToEncode := NewDense(2, 3, []Float{
		1, 2, 3,
		4, 5, 6,
	})

	var buf bytes.Buffer

	enc := gob.NewEncoder(&buf)
	err := enc.Encode(matrixToEncode)
	if err != nil {
		t.Fatal(err)
	}

	var decodedMatrix *Dense

	dec := gob.NewDecoder(&buf)
	err = dec.Decode(&decodedMatrix)
	if err != nil {
		t.Fatal(err)
	}

	assert.NotNil(t, decodedMatrix)
	assert.Equal(t, 2, decodedMatrix.rows)
	assert.Equal(t, 3, decodedMatrix.cols)
	assert.Equal(t, 6, decodedMatrix.size)
	assert.Equal(t, []Float{1, 2, 3, 4, 5, 6}, decodedMatrix.data)
	assert.Nil(t, decodedMatrix.viewOf)
	assert.False(t, decodedMatrix.fromPool)
}

func TestSparse_Gob(t *testing.T) {
	matrixToEncode := NewSparse(2, 3, []Float{
		1, 0, 3,
		0, 5, 0,
	})

	var buf bytes.Buffer

	enc := gob.NewEncoder(&buf)
	err := enc.Encode(matrixToEncode)
	if err != nil {
		t.Fatal(err)
	}

	var decodedMatrix *Dense

	dec := gob.NewDecoder(&buf)
	err = dec.Decode(&decodedMatrix)
	if err != nil {
		t.Fatal(err)
	}

	assert.NotNil(t, decodedMatrix)
	assert.Equal(t, 2, decodedMatrix.Rows())
	assert.Equal(t, 3, decodedMatrix.Columns())
	assert.Equal(t, []Float{1, 0, 3, 0, 5, 0}, decodedMatrix.Data())
}
