// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wordpiecetokenizer

import (
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/basetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
)

const (
	DefaultClassToken        = "[CLS]"
	DefaultSequenceSeparator = "[SEP]"
	DefaultUnknownToken      = "[UNK]"
	DefaultMaskToken         = "[MASK]"
	DefaultSplitPrefix       = "##"
	DefaultMaxWordChars      = 100
)

var DefaultNeverSplit = []string{
	DefaultClassToken,
	DefaultSequenceSeparator,
	DefaultUnknownToken,
	DefaultMaskToken,
}

var _ tokenizers.Tokenizer = &WordPieceTokenizer{}

// WordPieceTokenizer is a tokenizer that breaks tokens into sub-word units based on a supplied vocabulary.
// See https://arxiv.org/pdf/1609.08144.pdf Section 4.1 for details.
// WordPieceTokenizers uses BaseTokenizer to preprocess the input text.
type WordPieceTokenizer struct {
	baseTokenizer *basetokenizer.BaseTokenizer
	vocabulary    *vocabulary.Vocabulary
	unkToken      string
	splitPrefix   string
	maxWordChars  int
	neverSplit    []string
}

func New(vocabulary *vocabulary.Vocabulary) *WordPieceTokenizer {
	return &WordPieceTokenizer{
		baseTokenizer: basetokenizer.New(),
		vocabulary:    vocabulary,
		unkToken:      DefaultUnknownToken,
		splitPrefix:   DefaultSplitPrefix,
		maxWordChars:  DefaultMaxWordChars,
		neverSplit:    DefaultNeverSplit,
	}
}

// Tokenize converts the input text to a slice of words or sub-words token units based on the supplied vocabulary.
// The resulting tokens preserve the alignment with the portion of the original text they belong to.
func (t *WordPieceTokenizer) Tokenize(text string) []tokenizers.StringOffsetsPair {
	return t.WordPieceTokenize(t.baseTokenizer.Tokenize(text))
}

// WordPieceTokenize transforms the input token in a new slice of words or sub-words units based on the supplied vocabulary.
// The resulting tokens preserve the alignment with the portion of the original text they belong to.
func (t *WordPieceTokenizer) WordPieceTokenize(tokens []tokenizers.StringOffsetsPair) []tokenizers.StringOffsetsPair {
	outputTokens := make([]tokenizers.StringOffsetsPair, 0)

	for _, stringOffsetsPair := range tokens {
		token := stringOffsetsPair.String
		initialOffsets := stringOffsetsPair.Offsets
		characters := []rune(token)

		if len(characters) > t.maxWordChars {
			if _, exists := t.vocabulary.Id(t.unkToken); !exists {
				panic("Missing unk-token")
			}
			outputTokens = append(outputTokens, tokenizers.StringOffsetsPair{
				String:  t.unkToken,
				Offsets: initialOffsets,
			})
			continue
		}

		isBad := false
		start := 0
		subTokens := make([]tokenizers.StringOffsetsPair, 0)

		for start < len(characters) {
			end := len(characters)
			var curStrToken tokenizers.StringOffsetsPair
			found := false

			for start < end {
				subStr := string(characters[start:end])
				if start > 0 {
					subStr = t.splitPrefix + subStr
				}

				if _, exists := t.vocabulary.Id(subStr); exists {
					found = true
					curStrToken.String = subStr
					curStrToken.Offsets = tokenizers.OffsetsType{
						Start: initialOffsets.Start + start,
						End:   initialOffsets.Start + end,
					}
					break
				}
				end -= 1
			}
			if !found {
				isBad = true
				break
			}
			subTokens = append(subTokens, curStrToken)
			start = end
		}

		if isBad {
			if _, exists := t.vocabulary.Id(t.unkToken); !exists {
				panic("Missing unk-token")
			}
			outputTokens = append(outputTokens, tokenizers.StringOffsetsPair{
				String:  t.unkToken,
				Offsets: initialOffsets,
			})
		} else {
			outputTokens = append(outputTokens, subTokens...)
		}
	}
	return outputTokens
}

// IsDefaultSpecial return whether the word matches a special token, or not.
func IsDefaultSpecial(word string) bool {
	switch word {
	case DefaultUnknownToken, DefaultClassToken, DefaultSequenceSeparator, DefaultMaskToken:
		return true
	default:
		return false
	}
}
