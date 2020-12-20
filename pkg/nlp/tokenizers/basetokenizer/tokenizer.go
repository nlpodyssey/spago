// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package basetokenizer provides an implementation of a very simple tokenizer that splits by
// white-spaces (and alike) and punctuation symbols.
// Please note that abbreviations, real numbers, apostrophes and other expressions are tokenized
// without any linguistic criteria. It makes disasters on URLs, emails, etc.
package basetokenizer

import (
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"unicode"
)

// Ascii punctuation characters range
var asciiPunctuation = &unicode.RangeTable{
	R16: []unicode.Range16{
		{0x0021, 0x002f, 1}, // 33-47
		{0x003a, 0x0040, 1}, // 58-64
		{0x005b, 0x0060, 1}, // 91-96
		{0x007b, 0x007e, 1}, // 123-126
	},
	LatinOffset: 4, // All less than 0x00FF
}

var _ tokenizers.Tokenizer = &BaseTokenizer{}

// BaseTokenizer is a straightforward tokenizer implementation, which
// splits by whitespace and punctuation characters.
type BaseTokenizer struct {
	specialWords map[string]bool
}

// Option allows to configure a new BaseTokenizer with your specific needs.
type Option func(*BaseTokenizer)

// RegisterSpecialWords is an option to register a special word.
func RegisterSpecialWords(specialWords ...string) Option {
	return func(f *BaseTokenizer) {
		for _, word := range specialWords {
			f.specialWords[word] = true
		}
	}
}

// New returns a new base tokenizer ready to use.
func New(opts ...Option) *BaseTokenizer {
	t := &BaseTokenizer{
		specialWords: make(map[string]bool),
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Tokenize converts the input text to a slice of tokens, where each token is a white-separated word,
// a number or a punctuation sign.
// The resulting tokens preserve the alignment with the portion of the original text they belong to.
func (t *BaseTokenizer) Tokenize(text string) []tokenizers.StringOffsetsPair {
	splitTokens := make([]tokenizers.StringOffsetsPair, 0)
	spaceTokens := t.splitOn(text, isWhitespace, false)

	for _, spaceToken := range spaceTokens {
		if _, isSpecial := t.specialWords[spaceToken.String]; isSpecial {
			splitTokens = append(splitTokens, spaceToken)
			continue // TODO: this is temporary solution to don't split special tokens further; improve it.
		}

		puncTokens := t.splitOn(spaceToken.String, isPunctuation, true)
		for _, puncToken := range puncTokens {
			splitTokens = append(splitTokens, tokenizers.StringOffsetsPair{
				String: puncToken.String,
				Offsets: tokenizers.OffsetsType{
					Start: spaceToken.Offsets.Start + puncToken.Offsets.Start,
					End:   spaceToken.Offsets.Start + puncToken.Offsets.End,
				},
			})
		}
	}
	return splitTokens
}

// splitOn splits the given string as the `shouldSplit` predicate dictates.
// It keeps track of the offsets.
func (t *BaseTokenizer) splitOn(text string, shouldSplit func(rune) bool, includeSplitToken bool) []tokenizers.StringOffsetsPair {
	words := make([]tokenizers.StringOffsetsPair, 0)
	word := make([]rune, 0)
	offset := 0
	for _, r := range text {
		if shouldSplit(r) {
			wordLen := len(word)
			if wordLen > 0 {
				words = append(words, tokenizers.StringOffsetsPair{
					String:  string(word),
					Offsets: tokenizers.OffsetsType{Start: offset - wordLen, End: offset},
				})
				word = make([]rune, 0, cap(word))
			}
			if includeSplitToken {
				words = append(words, tokenizers.StringOffsetsPair{
					String:  string(r),
					Offsets: tokenizers.OffsetsType{Start: offset, End: offset + 1},
				})
			}
		} else {
			word = append(word, r)
		}
		offset++
	}

	// Don't forget the potential last word
	wordLen := len(word)
	if wordLen > 0 {
		words = append(words, tokenizers.StringOffsetsPair{
			String:  string(word),
			Offsets: tokenizers.OffsetsType{Start: offset - wordLen, End: offset},
		})
	}
	return words
}

// IsWhitespace checks whether rune c is a BERT whitespace character
func isWhitespace(r rune) bool {
	switch r {
	case ' ':
		return true
	case '\t':
		return true
	case '\n':
		return true
	case '\r':
		return true
	}
	return unicode.Is(unicode.Zs, r)
}

func isPunctuation(r rune) bool {
	return unicode.In(r, asciiPunctuation, unicode.P) && r != '-' // TODO: use regex or unicode.In()
}
