// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kvdb

import (
	"bytes"
	"encoding/gob"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"io/ioutil"
	"os"
	"testing"
)

func TestKeyValueDB_ReadOnlyAndForceNew(t *testing.T) {
	t.Parallel()

	dir := newTempDir(t, "spago-kvdb-test-")
	defer os.RemoveAll(dir)

	// Create a new DB and write something
	db1 := NewDefaultKeyValueDB(Config{Path: dir, ReadOnly: false, ForceNew: true})

	err := db1.Put([]byte{1}, []byte{2})
	require.Nil(t, err)

	err = db1.Close()
	require.Nil(t, err)

	// Reopen the same DB in read-only mode
	db2 := NewDefaultKeyValueDB(Config{Path: dir, ReadOnly: true, ForceNew: false})

	value, ok, err := db2.Get([]byte{1})
	require.Nil(t, err)
	assert.True(t, ok)
	assert.Equal(t, []byte{2}, value)

	err = db2.Put([]byte{3}, []byte{4})
	assert.NotNil(t, err)
	value, ok, err = db2.Get([]byte{3})
	require.Nil(t, err)
	assert.False(t, ok)
	assert.Nil(t, value)

	err = db2.Close()
	require.Nil(t, err)

	// Reopen a new DB reusing the same folder and forcing cleanup
	db3 := NewDefaultKeyValueDB(Config{Path: dir, ReadOnly: false, ForceNew: true})

	value, ok, err = db3.Get([]byte{1})
	require.Nil(t, err)
	assert.False(t, ok)
	assert.Nil(t, value)

	err = db3.Close()
	require.Nil(t, err)
}

func TestKeyValueDB_DropAll(t *testing.T) {
	t.Parallel()

	dir := newTempDir(t, "spago-kvdb-test-")
	defer os.RemoveAll(dir)

	db := NewDefaultKeyValueDB(Config{Path: dir, ReadOnly: false, ForceNew: true})
	defer db.Close()

	err := db.Put([]byte{1}, []byte{2})
	require.Nil(t, err)
	err = db.Put([]byte{3}, []byte{4})
	require.Nil(t, err)

	keys, err := db.Keys()
	require.Nil(t, err)
	assert.NotEmpty(t, keys)

	err = db.DropAll()
	require.Nil(t, err)

	keys, err = db.Keys()
	require.Nil(t, err)
	assert.Empty(t, keys)
}

func TestKeyValueDB_Keys(t *testing.T) {
	t.Parallel()

	dir := newTempDir(t, "spago-kvdb-test-")
	defer os.RemoveAll(dir)

	db := NewDefaultKeyValueDB(Config{Path: dir, ReadOnly: false, ForceNew: true})
	defer db.Close()

	err := db.Put([]byte{1}, []byte{2})
	require.Nil(t, err)

}

func TestKeyValueDB_PutAndGet(t *testing.T) {
	t.Parallel()

	dir := newTempDir(t, "spago-kvdb-test-")
	defer os.RemoveAll(dir)

	db := NewDefaultKeyValueDB(Config{Path: dir, ReadOnly: false, ForceNew: true})
	defer db.Close()

	err := db.Put([]byte{1}, []byte{2})
	require.Nil(t, err)

	// Existing key
	value, ok, err := db.Get([]byte{1})
	require.Nil(t, err)
	assert.True(t, ok)
	assert.Equal(t, []byte{2}, value)

	// Nonexistent key
	value, ok, err = db.Get([]byte{9})
	require.Nil(t, err)
	assert.False(t, ok)
	assert.Nil(t, value)
}

func TestKeyValueDB_Gob(t *testing.T) {
	t.Parallel()

	var buf bytes.Buffer

	// Setup first DB
	dir1 := newTempDir(t, "spago-kvdb-test-1-")
	defer os.RemoveAll(dir1)

	db1 := NewDefaultKeyValueDB(Config{Path: dir1, ReadOnly: false, ForceNew: true})
	defer db1.Close()
	err := db1.Put([]byte{42}, []byte{1})
	require.Nil(t, err)

	// Setup second DB
	dir2 := newTempDir(t, "spago-kvdb-test-2-")
	defer os.RemoveAll(dir2)

	db2 := NewDefaultKeyValueDB(Config{Path: dir2, ReadOnly: false, ForceNew: true})
	defer db2.Close()
	err = db2.Put([]byte{42}, []byte{2})
	require.Nil(t, err)

	// Encode first DB
	enc := gob.NewEncoder(&buf)
	err = enc.Encode(&db1)
	require.Nil(t, err)

	// Decode to second DB
	dec := gob.NewDecoder(&buf)
	err = dec.Decode(&db2)
	require.Nil(t, err)

	// First DB: still points to dir1 and contains 42 => 1
	assert.Equal(t, dir1, db1.Config.Path)
	v1, _, err := db1.Get([]byte{42})
	require.Nil(t, err)
	assert.Equal(t, []byte{1}, v1)

	// Second DB: it should be unmodified, pointing to dir2 and containing 42 => 2
	assert.Equal(t, dir2, db2.Config.Path)
	v2, _, err := db2.Get([]byte{42})
	require.Nil(t, err)
	assert.Equal(t, []byte{2}, v2)
}

func newTempDir(t *testing.T, pattern string) string {
	t.Helper()
	dir, err := ioutil.TempDir("", pattern)
	require.Nil(t, err)
	return dir
}
