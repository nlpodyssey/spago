// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tokenizers is an interim solution while developing `gotokenizers` (https://github.com/nlpodyssey/gotokenizers).
// APIs and implementations may be subject to frequent refactoring.
package tokenizers

// Tokenizer is implemented by any value that has the Tokenize method.
type Tokenizer interface {
	Tokenize(text string) []StringOffsetsPair
}

// StringOffsetsPair represents a string value paired with offsets bounds.
// It usually represents a token string and its offsets positions in the
// original string.
type StringOffsetsPair struct {
	String  string
	Offsets OffsetsType
}

// OffsetsType represents a (start, end) offsets pair.
// It usually represents a lower inclusive index position, and an upper
// exclusive position.
type OffsetsType struct {
	Start int
	End   int
}

// GetStrings returns a sequence of string values from the given slice
// of StringOffsetsPair.
func GetStrings(tokens []StringOffsetsPair) []string {
	result := make([]string, len(tokens))
	for i, stringOffsetsPair := range tokens {
		result[i] = stringOffsetsPair.String
	}
	return result
}

// GetOffsets returns a sequence of offsets values from the given slice
// of StringOffsetsPair.
func GetOffsets(tokens []StringOffsetsPair) []OffsetsType {
	result := make([]OffsetsType, len(tokens))
	for i, stringOffsetsPair := range tokens {
		result[i] = stringOffsetsPair.Offsets
	}
	return result
}
