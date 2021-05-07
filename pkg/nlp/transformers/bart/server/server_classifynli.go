// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/head/sequenceclassification"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/tasks"
	"time"
)

func (s *Server) classifyNLI(
	text string,
	hypothesisTemplate string,
	candidateLabels []string,
	multiClass bool,
) (*tasks.ClassifyResponse, error) {
	start := time.Now()

	task := tasks.BartForZeroShotClassification{
		Model:     s.model.(*sequenceclassification.Model),
		Tokenizer: s.bpeTokenizer,
	}

	result, err := task.Classify(
		text,
		hypothesisTemplate,
		candidateLabels,
		multiClass,
	)
	if err != nil {
		return nil, err
	}

	result.Took = time.Since(start).Milliseconds()
	return result, nil
}
