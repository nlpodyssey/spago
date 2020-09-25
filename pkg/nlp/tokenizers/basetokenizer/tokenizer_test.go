// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package basetokenizer

import (
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"testing"
)

func TestBaseTokenizer_Tokenize(t *testing.T) {
	run := func(s string, expected []tokenizers.StringOffsetsPair) {
		t.Run(s, func(t *testing.T) {
			tokenizer := &BaseTokenizer{}
			actual := tokenizer.Tokenize(s)

			if !stringOffsetsPairEqual(actual, expected) {
				t.Errorf("expected %v, got %v", expected, actual)
			}
		})
	}

	run("Hey friend! \n \t How are you?!?", []tokenizers.StringOffsetsPair{
		{String: "Hey", Offsets: tokenizers.OffsetsType{End: 3}},
		{String: "friend", Offsets: tokenizers.OffsetsType{Start: 4, End: 10}},
		{String: "!", Offsets: tokenizers.OffsetsType{Start: 10, End: 11}},
		{String: "How", Offsets: tokenizers.OffsetsType{Start: 16, End: 19}},
		{String: "are", Offsets: tokenizers.OffsetsType{Start: 20, End: 23}},
		{String: "you", Offsets: tokenizers.OffsetsType{Start: 24, End: 27}},
		{String: "?", Offsets: tokenizers.OffsetsType{Start: 27, End: 28}},
		{String: "!", Offsets: tokenizers.OffsetsType{Start: 28, End: 29}},
		{String: "?", Offsets: tokenizers.OffsetsType{Start: 29, End: 30}},
	})

	run("  foo  ", []tokenizers.StringOffsetsPair{
		{String: "foo", Offsets: tokenizers.OffsetsType{Start: 2, End: 5}},
	})

	run("  .  ", []tokenizers.StringOffsetsPair{
		{String: ".", Offsets: tokenizers.OffsetsType{Start: 2, End: 3}},
	})

	run("", []tokenizers.StringOffsetsPair{})
	run("   ", []tokenizers.StringOffsetsPair{})
}

func stringOffsetsPairEqual(a, b []tokenizers.StringOffsetsPair) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Offsets != b[i].Offsets || a[i].String != b[i].String {
			return false
		}
	}
	return true
}

func TestBaseTokenizer_isWhitespace(t *testing.T) {
	for _, test := range []struct {
		char  rune
		valid bool
	}{
		{' ', true},
		{'\t', true},
		{'\r', true},
		{'\n', true},
		{'\u00A0', true},
		{'A', false},
		{'-', false},
	} {
		if isWhitespace(test.char) != test.valid {
			t.Errorf("invalid whitespace: %U, %t", test.char, test.valid)
		}
	}
}

func TestBaseTokenizer_isPunctuation(t *testing.T) {
	for _, test := range []struct {
		char  rune
		valid bool
	}{
		{'-', false}, // TODO: find a robust solution to handle the hyphen
		{'$', true},
		{'`', true},
		{'.', true},
		{'A', false},
		{' ', false},
	} {
		if isPunctuation(test.char) != test.valid {
			t.Errorf("invalid punctuation: %U, %t", test.char, test.valid)
		}
	}
}
