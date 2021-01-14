// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kvdb

import (
	"bytes"
	"encoding/gob"
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	"os"
	"testing"
)

func TestKeyValueDB_Gob(t *testing.T) {
	var buf bytes.Buffer

	// Setup first DB
	dir1 := newTempDir(t, "spago-kvdb-test-1-")
	defer os.RemoveAll(dir1)

	db1 := NewDefaultKeyValueDB(Config{Path: dir1, ReadOnly: false, ForceNew: true})
	defer db1.Close()
	err := db1.Put([]byte{42}, []byte{1})
	if err != nil {
		t.Fatal(err)
	}

	// Setup second DB
	dir2 := newTempDir(t, "spago-kvdb-test-2-")
	defer os.RemoveAll(dir2)

	db2 := NewDefaultKeyValueDB(Config{Path: dir2, ReadOnly: false, ForceNew: true})
	defer db2.Close()
	err = db2.Put([]byte{42}, []byte{2})
	if err != nil {
		t.Fatal(err)
	}

	// Encode first DB
	enc := gob.NewEncoder(&buf)
	err = enc.Encode(&db1)
	if err != nil {
		t.Fatal(err)
	}

	// Decode to second DB
	dec := gob.NewDecoder(&buf)
	err = dec.Decode(&db2)
	if err != nil {
		t.Fatal(err)
	}

	// First DB: still points to dir1 and contains 42 => 1
	assert.Equal(t, dir1, db1.Config.Path)
	v1, _, err := db1.Get([]byte{42})
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, []byte{1}, v1)

	// Second DB: it should be unmodified, pointing to dir2 and containing 42 => 2
	assert.Equal(t, dir2, db2.Config.Path)
	v2, _, err := db2.Get([]byte{42})
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, []byte{2}, v2)
}

func newTempDir(t *testing.T, pattern string) string {
	t.Helper()
	dir, err := ioutil.TempDir("", pattern)
	if err != nil {
		t.Fatal(err)
	}
	return dir
}
