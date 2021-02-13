// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sentencepiece

import (
	"fmt"
	"github.com/nlpodyssey/gotokenizers/vocabulary"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/sentencepiece/internal/sentencepiece"
	"path/filepath"
	"strings"
)

const defaultUnknownToken = "<unk>"
const defaultSeparator = "▁"

// SentencePieceTokenizer is a Sentence Piece tokenizer.
type SentencePieceTokenizer struct {
	sp    *sentencepiece.Sentencepiece
	vocab *vocabulary.Vocabulary
}

// NewFromModelFolder returns a new SentencePieceTokenizer.
func NewFromModelFolder(path string, lowercase bool) (*SentencePieceTokenizer, error) {
	vocabFilename := filepath.Join(path, "vocab.json")
	vocab, err := vocabulary.FromJSONFile(vocabFilename)
	if err != nil {
		return nil, fmt.Errorf("loading vocabulary from file %s: %w", vocabFilename, err)
	}

	spmFilename := filepath.Join(path, "source.spm")
	sp, err := sentencepiece.NewSentencepieceFromFile(spmFilename, lowercase)
	if err != nil {
		return nil, fmt.Errorf("loading sentence-piece from file %s: %w", spmFilename, err)
	}

	return &SentencePieceTokenizer{
		sp:    &sp,
		vocab: vocab,
	}, nil
}

// Tokenize performs sentence-piece tokenization.
func (t *SentencePieceTokenizer) Tokenize(text string) []string {
	tokens := t.sp.Tokenize(text)

	result := make([]string, len(tokens))
	for i, token := range tokens {
		result[i] = token.Text
	}
	return result
}

// TokensToIDs returns a list of token IDs from a list of string tokens.
// It panics if a token is not found in the vocabulary and no unknown token is found.
func (t *SentencePieceTokenizer) TokensToIDs(tokens []string) []int {
	ids := make([]int, len(tokens))
	for i, token := range tokens {
		var ok bool
		ids[i], ok = t.vocab.GetID(token)
		if !ok {
			ids[i], ok = t.vocab.GetID(defaultUnknownToken)
			if !ok {
				panic(fmt.Errorf("unknown token ID not found for token %#v", token))
			}
		}
	}
	return ids
}

// IDsToTokens returns a list of string terms from a list of token IDs.
// It panics if a token is not found in the vocabulary.
func (t *SentencePieceTokenizer) IDsToTokens(ids []int) []string {
	tokens := make([]string, len(ids))
	for i, id := range ids {
		var ok bool
		tokens[i], ok = t.vocab.GetString(id)
		if !ok {
			panic(fmt.Errorf("unknown token string value for ID %d", id))
		}
	}
	return tokens
}

// Detokenize flatten and merges a list of tokens into a single string.
func (t *SentencePieceTokenizer) Detokenize(tokens []string) string {
	var sb strings.Builder

	for i, token := range tokens {
		if strings.HasPrefix(token, defaultSeparator) {
			if i > 0 {
				sb.WriteByte(' ')
			}
			token = token[len(defaultSeparator):]
		}
		sb.WriteString(token)
	}

	return sb.String()
}
