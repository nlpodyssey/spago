// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpetokenizer

import (
	"fmt"
	"github.com/nlpodyssey/gotokenizers/encodings"
	"github.com/nlpodyssey/gotokenizers/models"
	"github.com/nlpodyssey/gotokenizers/models/bpemodel"
	"github.com/nlpodyssey/gotokenizers/normalizedstring"
	"github.com/nlpodyssey/gotokenizers/pretokenizedstring"
	"github.com/nlpodyssey/gotokenizers/pretokenizers/bytelevelpretokenizer"
	"github.com/nlpodyssey/gotokenizers/strutils"
	"github.com/nlpodyssey/gotokenizers/vocabulary"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"path/filepath"
)

// var _ tokenizers.Tokenizer = &BPETokenizer{} // TODO: update Tokenizer interface to return errors

// BPETokenizer is a higher-level tokenizer, which includes byte-level pre-tokenization.
type BPETokenizer struct {
	preTokenizer *bytelevelpretokenizer.ByteLevelPreTokenizer
	model        *bpemodel.BPEModel
}

// New returns a new BPETokenizer.
func New(
	preTokenizer *bytelevelpretokenizer.ByteLevelPreTokenizer,
	model *bpemodel.BPEModel,
) *BPETokenizer {
	return &BPETokenizer{
		preTokenizer: preTokenizer,
		model:        model,
	}
}

const (
	defaultCacheCapacity           = 0
	defaultDropout                 = 0.0
	defaultUnknownToken            = ""
	defaultContinuingSubwordPrefix = ""
	defaultEndOfWordSuffix         = ""
	defaultPrefixSpaceEnabled      = false
	defaultOffsetsTrimmingEnabled  = true
	defaultUnknownFusionEnabled    = false
)

// NewFromModelFolder returns a new BPETokenizer built from a
// pre-trained Roberta-compatible model, given the path to the
// folder containing the separate model and configuration files.
func NewFromModelFolder(path string) (*BPETokenizer, error) {
	vocabularyFilename := filepath.Join(path, "vocab.json")
	vocab, err := vocabulary.FromJSONFile(vocabularyFilename)
	if err != nil {
		return nil, fmt.Errorf("loading vocabulary from file %s: %w", vocabularyFilename, err)
	}

	mergesFilename := filepath.Join(path, "merges.txt")
	merges, err := bpemodel.MergeMapFromFile(
		mergesFilename,
		vocab,
		len(defaultContinuingSubwordPrefix),
	)
	if err != nil {
		return nil, fmt.Errorf("loading merges from file %s: %w", mergesFilename, err)
	}

	preTokenizer := bytelevelpretokenizer.New(
		bytelevelpretokenizer.DefaultSplittingRegexp,
		defaultPrefixSpaceEnabled,
		defaultOffsetsTrimmingEnabled,
	)

	model := bpemodel.New(
		vocab,
		merges,
		defaultCacheCapacity,
		defaultDropout,
		defaultUnknownToken,
		defaultContinuingSubwordPrefix,
		defaultEndOfWordSuffix,
		defaultUnknownFusionEnabled,
	)

	return New(preTokenizer, model), nil
}

// Tokenize performs byte-level pre-tokenization and BPE tokenization.
func (t *BPETokenizer) Tokenize(text string) ([]tokenizers.StringOffsetsPair, error) {
	pts := pretokenizedstring.FromString(text)

	err := t.preTokenizer.PreTokenize(pts)
	if err != nil {
		return nil, fmt.Errorf("BPETokenizer PreTokenize for %s: %w", text, err)
	}

	err = pts.Tokenize(
		func(ns *normalizedstring.NormalizedString) ([]models.Token, error) {
			return t.model.Tokenize(ns.Get())
		},
	)
	if err != nil {
		return nil, fmt.Errorf("BPETokenizer Tokenize for %s: %w", text, err)
	}

	converter := strutils.NewBytesToRuneOffsetConverter(text)
	result := make([]tokenizers.StringOffsetsPair, 0)

	for _, split := range pts.Splits() {
		ns := split.NormalizedString
		nsOffsets := ns.OriginalOffsets()

		for _, token := range *split.Tokens {
			origOffset, ok := ns.CoerceRangeToOriginal(
				normalizedstring.NewNormalizedRange(
					token.Offsets.Start,
					token.Offsets.End,
				),
			)
			if !ok {
				return nil, fmt.Errorf("BPETokenizer range coercion for %s: %w", text, err)
			}

			byteOffsets := strutils.ByteOffsets{
				Start: nsOffsets.Start + origOffset.Start(),
				End:   nsOffsets.Start + origOffset.End(),
			}

			runeOffsets := converter.Convert(byteOffsets)

			result = append(result, tokenizers.StringOffsetsPair{
				String: token.Value,
				Offsets: tokenizers.OffsetsType{
					Start: runeOffsets.Start,
					End:   runeOffsets.End,
				},
			})
		}
	}

	return result, nil
}

// Encode converts a text into an encoded tokens representation useful for Transformer architectures.
// It tokenizes using byte-level pre-tokenization and BPE tokenization.
func (t *BPETokenizer) Encode(text string) (*encodings.Encoding, error) {
	pts := pretokenizedstring.FromString(text)

	err := t.preTokenizer.PreTokenize(pts)
	if err != nil {
		return nil, fmt.Errorf("BPETokenizer PreTokenize for %s: %w", text, err)
	}

	err = pts.Tokenize(
		func(ns *normalizedstring.NormalizedString) ([]models.Token, error) {
			return t.model.Tokenize(ns.Get())
		},
	)
	if err != nil {
		return nil, fmt.Errorf("BPETokenizer Tokenize for %s: %w", text, err)
	}

	encoding, err := pts.IntoEncoding(0, 0)
	if err != nil {
		return nil, fmt.Errorf("BPETokenizer Encoding for %s: %w", text, err)
	}
	return encoding, nil
}
