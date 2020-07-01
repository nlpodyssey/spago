// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package grpcutils

import (
	"log"
	"net"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

// NewGRPCServer returns grpc.Server objects, optionally configured for TLS.
func NewGRPCServer(tlsDisable bool, tlsCert, tlsKey string) *grpc.Server {
	serverOptions := createServerOptions(tlsDisable, tlsCert, tlsKey)
	grpcServer := grpc.NewServer(serverOptions...)
	return grpcServer
}

func createServerOptions(tlsDisable bool, tlsCert, tlsKey string) []grpc.ServerOption {
	if tlsDisable {
		return []grpc.ServerOption{}
	}

	creds, err := credentials.NewServerTLSFromFile(tlsCert, tlsKey)
	if err != nil {
		log.Fatalf("failed to read TLS certs: %v\n", err)
	}

	return []grpc.ServerOption{
		grpc.Creds(creds),
	}
}

// RunGRPCServer listens on the given address and serves the given *grpc.Server,
// and blocks until done.
func RunGRPCServer(grpcAddress string, grpcServer *grpc.Server) {
	listener := newListenerForGRPC(grpcAddress)
	log.Fatal(grpcServer.Serve(listener))
}

func newListenerForGRPC(grpcAddress string) net.Listener {
	result, err := net.Listen("tcp", grpcAddress)

	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	return result
}
