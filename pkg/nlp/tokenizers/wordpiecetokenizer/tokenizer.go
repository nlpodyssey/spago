// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wordpiecetokenizer

import (
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/basetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"strings"
)

const (
	// DefaultClassToken is the default class token value for the WordPiece tokenizer.
	DefaultClassToken = "[CLS]"
	// DefaultSequenceSeparator is the default sequence separator value for the WordPiece tokenizer.
	DefaultSequenceSeparator = "[SEP]"
	// DefaultUnknownToken is the default unknown token value for the WordPiece tokenizer.
	DefaultUnknownToken = "[UNK]"
	// DefaultMaskToken is the default mask token value for the WordPiece tokenizer.
	DefaultMaskToken = "[MASK]"
	// DefaultSplitPrefix is the default split prefix value for the WordPiece tokenizer.
	DefaultSplitPrefix = "##"
	// DefaultMaxWordChars is the default maximum word length for the WordPiece tokenizer.
	DefaultMaxWordChars = 100
)

var defaultNeverSplit = []string{
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

// New returns a new WordPieceTokenizer.
func New(vocabulary *vocabulary.Vocabulary) *WordPieceTokenizer {
	return &WordPieceTokenizer{
		baseTokenizer: basetokenizer.New(
			basetokenizer.RegisterSpecialWords(DefaultUnknownToken, DefaultClassToken, DefaultSequenceSeparator, DefaultMaskToken)),
		vocabulary:   vocabulary,
		unkToken:     DefaultUnknownToken,
		splitPrefix:  DefaultSplitPrefix,
		maxWordChars: DefaultMaxWordChars,
		neverSplit:   defaultNeverSplit,
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
			if _, exists := t.vocabulary.ID(t.unkToken); !exists {
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

				if _, exists := t.vocabulary.ID(subStr); exists {
					found = true
					curStrToken.String = subStr
					curStrToken.Offsets = tokenizers.OffsetsType{
						Start: initialOffsets.Start + start,
						End:   initialOffsets.Start + end,
					}
					break
				}
				end--
			}
			if !found {
				isBad = true
				break
			}
			subTokens = append(subTokens, curStrToken)
			start = end
		}

		if isBad {
			if _, exists := t.vocabulary.ID(t.unkToken); !exists {
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

// TokensRange represents an index offsets pair of a token.
type TokensRange struct {
	Start int
	End   int
}

// GroupPieces returns a list of tokens range each of which represents
// the start and the end index of the tokens that form a complete word.
func GroupPieces(tokens []tokenizers.StringOffsetsPair) []TokensRange {
	groups := make([]TokensRange, 0)
	for i, token := range tokens {
		if strings.HasPrefix(token.String, DefaultSplitPrefix) {
			groups[len(groups)-1].End = i
		} else {
			groups = append(groups, TokensRange{
				Start: i,
				End:   i,
			})
		}
	}
	return groups
}

// MakeOffsetPairsFromGroups creates a sequence tokenizers.StringOffsetsPair
// elements from the given groups.
func MakeOffsetPairsFromGroups(
	text string,
	tokens []tokenizers.StringOffsetsPair,
	groups []TokensRange,
) []tokenizers.StringOffsetsPair {
	outputTokens := make([]tokenizers.StringOffsetsPair, len(groups))
	for i, group := range groups {
		startToken, endToken := tokens[group.Start], tokens[group.End]
		outputTokens[i] = tokenizers.StringOffsetsPair{
			String: string([]rune(text)[startToken.Offsets.Start:endToken.Offsets.End]),
			Offsets: tokenizers.OffsetsType{
				Start: startToken.Offsets.Start,
				End:   endToken.Offsets.End,
			},
		}
	}
	return outputTokens
}
