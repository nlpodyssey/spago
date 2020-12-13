// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tokenizers is an interim solution while developing `gotokenizers` (https://github.com/nlpodyssey/gotokenizers).
// APIs and implementations may be subject to frequent refactoring.
package tokenizers

type Tokenizer interface {
	Tokenize(text string) []StringOffsetsPair
}

type StringOffsetsPair struct {
	String  string
	Offsets OffsetsType
}

type OffsetsType struct {
	Start int
	End   int
}

func GetStrings(tokens []StringOffsetsPair) []string {
	result := make([]string, len(tokens))
	for i, stringOffsetsPair := range tokens {
		result[i] = stringOffsetsPair.String
	}
	return result
}

func GetOffsets(tokens []StringOffsetsPair) []OffsetsType {
	result := make([]OffsetsType, len(tokens))
	for i, stringOffsetsPair := range tokens {
		result[i] = stringOffsetsPair.Offsets
	}
	return result
}
