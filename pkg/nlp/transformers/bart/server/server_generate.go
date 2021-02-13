// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/head/conditionalgeneration"
	"time"
)

func (s *Server) generate(text string) (*GenerateResponse, error) {
	start := time.Now()

	g := ag.NewGraph()
	defer g.Clear()

	proc := nn.Reify(nn.Context{Graph: g, Mode: nn.Inference}, s.model).(*conditionalgeneration.Model)
	bartConfig := proc.BART.Config

	tokens := s.spTokenizer.Tokenize(text)
	tokenIDs := s.spTokenizer.TokensToIDs(tokens)

	tokenIDs = append(tokenIDs, bartConfig.EosTokenID)

	rawGeneratedIDs := proc.Generate(tokenIDs)
	generatedIDs := s.stripBadTokens(rawGeneratedIDs, bartConfig)

	generatedTokens := s.spTokenizer.IDsToTokens(generatedIDs)
	generatedText := s.spTokenizer.Detokenize(generatedTokens)

	return &GenerateResponse{
		Text: generatedText,
		Took: time.Since(start).Milliseconds(),
	}, nil
}

func (s *Server) stripBadTokens(ids []int, bartConfig config.Config) []int {
	result := make([]int, 0, len(ids))
	for _, id := range ids {
		if id == bartConfig.EosTokenID || id == bartConfig.PadTokenID || id == bartConfig.BosTokenID ||
			id == bartConfig.DecoderStartTokenID {
			continue
		}
		result = append(result, id)
	}
	return result
}
