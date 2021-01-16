// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"io/ioutil"
	"os"
	"testing"
)

func TestFileSerialization(t *testing.T) {
	tempFile, err := ioutil.TempFile("", "spago-serialization-test-")
	require.Nil(t, err)
	fileName := tempFile.Name()
	defer os.Remove(fileName)

	type MyStruct struct {
		X int
		S *MyStruct
	}

	obj1 := MyStruct{
		X: 1,
		S: &MyStruct{X: 2},
	}

	err = SerializeToFile(fileName, obj1)
	require.Nil(t, err)

	var obj2 MyStruct
	err = DeserializeFromFile(fileName, &obj2)
	require.Nil(t, err)
	assert.Equal(t, obj1, obj2)
}

func TestCountLines(t *testing.T) {
	tempFile, err := ioutil.TempFile("", "spago-count-lines-test-")
	require.Nil(t, err)
	fileName := tempFile.Name()
	defer os.Remove(fileName)

	err = ioutil.WriteFile(fileName, []byte("foo\nbar\nbaz\n"), 0644)
	require.Nil(t, err)

	n, err := CountLines(fileName)
	require.Nil(t, err)
	assert.Equal(t, 3, n)
}
