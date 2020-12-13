// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vocabulary_test

import (
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
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
