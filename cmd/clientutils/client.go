// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package clientutils

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/urfave/cli"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"gopkg.in/yaml.v3"
)

// Flags returns a list of the common CLI flags for gRPC clients
// combined with a list of specific CLI command flags.
func Flags(address *string, tlsDisable *bool, output *string, cmdFlags []cli.Flag) []cli.Flag {
	grpcClientFlags := []cli.Flag{
		cli.StringFlag{
			Name:        "address",
			Value:       "127.0.0.1:1976",
			Destination: address,
		},
		cli.BoolFlag{
			Name:        "tls-disable ",
			Usage:       "Specifies that TLS is disabled.",
			Destination: tlsDisable,
		},
		cli.StringFlag{
			Name:        "o, output",
			Value:       "yaml",
			Usage:       "Output format. One of: json|yaml",
			Destination: output,
		},
	}

	return append(grpcClientFlags, cmdFlags...)
}

// UsageText returns the usage text for gRPC clients that may be appended to existing usage text.
func UsageText() string {
	return " [--address=<address>] [--tls-disable] [(-o|--output=)json|yaml]"
}

// VerifyFlags verifies the values of specific client flags such as `output`.
func VerifyFlags(outputFlag string) {
	if !strings.EqualFold(outputFlag, "json") && !strings.EqualFold(outputFlag, "yaml") {
		log.Fatalln("Unsupported output format")
	}
}

// Println prints the response using the desired format.
func Println(format string, resp interface{}) {
	if format == "json" {
		out, err := json.MarshalIndent(resp, "", "  ")
		if err != nil {
			log.Fatalln(err)
		}
		fmt.Println(string(out))
	} else {
		out, err := yaml.Marshal(resp)
		if err != nil {
			log.Fatalln(err)
		}
		fmt.Println(string(out))
	}
}

// OpenConnection returns a new grpc.ClientConn object. It blocks until
// a connection is made or the process timed out.
func OpenConnection(address string, tlsDisable bool) *grpc.ClientConn {
	if tlsDisable {
		conn, err := grpc.Dial(address, grpc.WithInsecure())
		if err != nil {
			log.Fatalln(err)
		}
		return conn
	}

	creds := credentials.NewTLS(&tls.Config{
		InsecureSkipVerify: true,
	})
	conn, err := grpc.Dial(address, grpc.WithTransportCredentials(creds))
	if err != nil {
		log.Fatalln(err)
	}
	return conn
}
