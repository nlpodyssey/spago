// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vocabulary_test

import (
	"bytes"
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestNew(t *testing.T) {
	items := []string{"word1", "word2", "word3"}
	voc := vocabulary.New(items)
	for i, item := range items {
		if id, _ := voc.ID(item); id != i {
			t.Error("The id doesn't match the expected value")
		}
	}
}

func TestVocabulary_LongestPrefix(t *testing.T) {
	items := []string{"a", "aa", "aaa", "bbbb"}
	voc := vocabulary.New(items)
	for _, test := range []struct {
		term   string
		prefix string
	}{
		{"", ""},
		{"aabb", "aa"},
		{"aaabbbb", "aaa"},
		{"bbb", ""},
		{"bbbbb", "bbbb"},
	} {
		sub := voc.LongestPrefix(test.term)
		if sub != test.prefix {
			t.Errorf("invalid longest prefix. Expected: %v, Found: %v", test.prefix, sub)
		}
	}
}

func TestVocabulary_Gob(t *testing.T) {
	terms := []string{"foo", "bar", "baz"}
	v1 := vocabulary.New(terms)

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(&v1)
	require.Nil(t, err)

	var v2 *vocabulary.Vocabulary

	dec := gob.NewDecoder(&buf)
	err = dec.Decode(&v2)
	require.Nil(t, err)
	assert.Equal(t, v1, v2)
}
