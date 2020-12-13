// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpetokenizer

import (
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"reflect"
	"testing"
)

func TestNewFromModelFolder(t *testing.T) {
	tokenizer, err := NewFromModelFolder("testdata/dummy-roberta-model")
	if err != nil {
		t.Fatal(err)
	}
	if tokenizer == nil {
		t.Fatal("expected *BPETokenizer, actual nil")
	}

	actual, _ := tokenizer.Tokenize("related unrelated")

	expected := []tokenizers.StringOffsetsPair{
		{
			String:  "related",
			Offsets: tokenizers.OffsetsType{Start: 0, End: 7},
		},
		{
			String:  "unrelated",
			Offsets: tokenizers.OffsetsType{Start: 7, End: 15},
		},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected:\n  %#v\nactual:\n  %#v\n", expected, actual)
	}
}
