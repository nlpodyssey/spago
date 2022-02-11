// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/head/sequenceclassification"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/tasks"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/tasks/zsc"
	"time"
)

func (s *Server[T]) classifyNLI(
	text string,
	hypothesisTemplate string,
	candidateLabels []string,
	multiClass bool,
) (*tasks.ClassifyResponse[T], error) {
	start := time.Now()

	task := zsc.BartForZeroShotClassification[T]{
		Model:     s.model.(*sequenceclassification.Model[T]),
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
