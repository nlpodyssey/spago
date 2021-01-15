// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syncmap

import (
	"bytes"
	"encoding/gob"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestSyncMap_Gob(t *testing.T) {
	var buf bytes.Buffer

	m1 := New()
	m1.Store("foo", "bar")

	err := gob.NewEncoder(&buf).Encode(&m1)
	require.Nil(t, err)

	m2 := New()
	m2.Store("foo", "qux")

	err = gob.NewDecoder(&buf).Decode(&m2)
	require.Nil(t, err)

	value, ok := m1.Load("foo")
	assert.True(t, ok)
	assert.Equal(t, "bar", value)

	value, ok = m2.Load("foo")
	assert.False(t, ok)
}
