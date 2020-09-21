// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"context"
	"encoding/json"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"net/http"
	"sort"
	"time"

	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
)

// ClassifyHandler handles a classify request over HTTP.
func (s *Server) ClassifyHandler(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "application/json")

	var body Body
	err := json.NewDecoder(req.Body).Decode(&body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result := s.classify(body.Text)
	_, pretty := req.URL.Query()["pretty"]
	response, err := Dump(result, pretty)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	_, err = w.Write(response)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

type ClassConfidencePair struct {
	Class      string  `json:"class"`
	Confidence float64 `json:"confidence"`
}

type ClassifyResponse struct {
	Class        string                `json:"class"`
	Confidence   float64               `json:"confidence"`
	Distribution []ClassConfidencePair `json:"distribution"`
	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}

// Predict handles a predict request over gRPC.
// TODO(evanmcclure@gmail.com) Reuse the gRPC message type for HTTP requests.
func (s *Server) Classify(_ context.Context, req *grpcapi.ClassifyRequest) (*grpcapi.ClassifyReply, error) {
	result := s.classify(req.GetText())
	return classificationFrom(result), nil
}

func classificationFrom(resp *ClassifyResponse) *grpcapi.ClassifyReply {
	distribution := make([]*grpcapi.ClassConfidencePair, len(resp.Distribution))
	for i, t := range resp.Distribution {
		distribution[i] = &grpcapi.ClassConfidencePair{
			Class:      t.Class,
			Confidence: t.Confidence,
		}
	}

	return &grpcapi.ClassifyReply{
		Class:        resp.Class,
		Confidence:   resp.Confidence,
		Distribution: distribution,
		Took:         resp.Took,
	}
}

// TODO: This method is too long; it needs to be refactored.
func (s *Server) classify(text string) *ClassifyResponse {
	start := time.Now()

	tokenizer := wordpiecetokenizer.New(s.model.Vocabulary)
	origTokens := tokenizer.Tokenize(text)
	tokenized := pad(tokenizers.GetStrings(origTokens))

	g := ag.NewGraph()
	defer g.Clear()
	proc := s.model.NewProc(g).(*Processor)
	proc.SetMode(nn.Inference)
	encoded := proc.Encode(tokenized)
	logits := proc.SequenceClassification(encoded)
	probs := f64utils.SoftMax(logits.Value().Data())
	best := f64utils.ArgMax(probs)
	class := s.model.Classifier.Config.Labels[best]

	distribution := make([]ClassConfidencePair, len(probs))
	for i := 0; i < len(probs); i++ {
		distribution[i] = ClassConfidencePair{
			Class:      s.model.Classifier.Config.Labels[i],
			Confidence: probs[i],
		}
	}

	sort.Slice(distribution, func(i, j int) bool {
		return distribution[i].Confidence > distribution[j].Confidence
	})

	return &ClassifyResponse{
		Class:        class,
		Confidence:   probs[best],
		Distribution: distribution,
		Took:         time.Since(start).Milliseconds(),
	}
}
