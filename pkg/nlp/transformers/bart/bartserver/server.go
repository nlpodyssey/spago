// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartserver

import (
	"context"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/bpetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/barthead"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartserver/grpcapi"
	"github.com/nlpodyssey/spago/pkg/utils/grpcutils"
)

// Server contains everything needed to run a BART server.
type ServerForSequenceClassification struct {
	model     *barthead.SequenceClassification
	tokenizer *bpetokenizer.BPETokenizer

	// UnimplementedBARTServer must be embedded to have forward compatible implementations for gRPC.
	grpcapi.UnimplementedBARTServer
}

func NewServer(
	model *barthead.SequenceClassification,
	tokenizer *bpetokenizer.BPETokenizer,
) *ServerForSequenceClassification {
	return &ServerForSequenceClassification{
		model:     model,
		tokenizer: tokenizer,
	}
}

// StartDefaultServer is used to start a basic BART gRPC server.
func (s *ServerForSequenceClassification) StartDefaultServer(grpcAddress, tlsCert, tlsKey string, tlsDisable bool) {
	grpcServer := grpcutils.NewGRPCServer(tlsDisable, tlsCert, tlsKey)
	grpcapi.RegisterBARTServer(grpcServer, s)
	grpcutils.RunGRPCServer(grpcAddress, grpcServer)
}

// Classify handles a classification request over gRPC.
func (s *ServerForSequenceClassification) Classify(_ context.Context, req *grpcapi.ClassifyRequest) (*grpcapi.ClassifyReply, error) {
	result := s.classify(req.GetText(), req.GetText2())
	return classificationFrom(result), nil
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

const (
	defaultStartSequenceTokenID = 0
	defaultEndSequenceTokenID   = 2
)

func getInputIDs(tokenizer *bpetokenizer.BPETokenizer, text, text2 string) []int {
	encoded, _ := tokenizer.Encode(text) // TODO: error handling
	inputIds := append(append([]int{defaultStartSequenceTokenID}, encoded.IDs...), defaultEndSequenceTokenID)
	if text2 != "" {
		encoded2, _ := tokenizer.Encode(text2) // TODO: error handling
		inputIds2 := append(append([]int{defaultEndSequenceTokenID}, encoded2.IDs...), defaultEndSequenceTokenID)
		inputIds = append(inputIds, inputIds2...)
	}
	return inputIds
}
