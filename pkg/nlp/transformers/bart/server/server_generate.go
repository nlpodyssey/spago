// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/head/conditionalgeneration"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/tasks"
	"time"
)

func (s *Server) generate(text string) (*GenerateResponse, error) {
	start := time.Now()

	task := tasks.BartForConditionalGeneration{
		Model:     s.model.(*conditionalgeneration.Model),
		Tokenizer: s.spTokenizer,
	}

	generated, err := task.Generate(text)
	if err != nil {
		return nil, err
	}

	return &GenerateResponse{
		Text: generated,
		Took: time.Since(start).Milliseconds(),
	}, nil
}
