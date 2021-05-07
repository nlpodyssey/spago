// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
)

// PredictHandler handles a predict request over HTTP.
func (s *Server) PredictHandler(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "application/json")

	var body Body
	err := json.NewDecoder(req.Body).Decode(&body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	start := time.Now()
	result := Response{
		Tokens: s.model.PredictMLM(body.Text),
		Took:   time.Since(start).Milliseconds(),
	}
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

// Predict handles a predict request over gRPC.
// TODO(evanmcclure@gmail.com) Reuse the gRPC message type for HTTP requests.
func (s *Server) Predict(ctx context.Context, req *grpcapi.PredictRequest) (*grpcapi.PredictReply, error) {
	start := time.Now()
	result := s.model.PredictMLM(req.GetText())
	return &grpcapi.PredictReply{
		Tokens: tokensFrom(result),
		Took:   time.Since(start).Milliseconds(),
	}, nil
}
