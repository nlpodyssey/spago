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
	return nil, nil // TODO: implement classification algorithm
}
