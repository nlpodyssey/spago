// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/bpetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/barthead"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartserver"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/huggingface"
	"github.com/nlpodyssey/spago/pkg/utils/httputils"
	"github.com/urfave/cli"
	"log"
	"os"
	"os/user"
	"path"
	"path/filepath"
)

func newServerCommandFor(app *BartApp) cli.Command {
	return cli.Command{
		Name:        "server",
		Usage:       "Run the " + programName + " as gRPC/HTTP server.",
		Description: "Run the " + programName + " indicating the model path (NOT the model file).",
		Flags:       newServerCommandFlagsFor(app),
		Action:      newServerCommandActionFor(app),
	}
}

func newServerCommandFlagsFor(app *BartApp) []cli.Flag {
	usr, err := user.Current()
	if err != nil {
		log.Fatal(err)
	}

	return []cli.Flag{
		cli.StringFlag{
			Name:        "grpc-address",
			Usage:       "Changes the bind address of the gRPC server.",
			Value:       "0.0.0.0:1976",
			Destination: &app.grpcAddress,
		},
		cli.StringFlag{
			Name:        "address",
			Usage:       "Changes the bind address of the HTTP JSON server.",
			Value:       "0.0.0.0:1987",
			Destination: &app.address,
		},
		cli.StringFlag{
			Name:        "repo",
			Usage:       "Specifies the path to the models.",
			EnvVar:      "SPAGO_REPO",
			Value:       path.Join(usr.HomeDir, ".spago"),
			Destination: &app.repo,
		},
		cli.StringFlag{
			Name:        "model, m",
			Required:    true,
			EnvVar:      "SPAGO_MODEL",
			Usage:       "Specifies the model name.",
			Destination: &app.model,
		},
		cli.StringFlag{
			Name:        "tls-cert-file",
			Usage:       "Specifies the path of the TLS certificate file.",
			Value:       "/etc/ssl/certs/spago/server.crt",
			Destination: &app.tlsCert,
		},
		cli.StringFlag{
			Name:        "tls-key-file",
			Usage:       "Specifies the path of the private key for the certificate.",
			Value:       "/etc/ssl/certs/spago/server.key",
			Destination: &app.tlsKey,
		},
		cli.BoolFlag{
			Name:        "tls-disable",
			Usage:       "Specifies that TLS is disabled.",
			Destination: &app.tlsDisable,
		},
		cli.IntFlag{
			Name:        "timeout",
			Usage:       "Server read, write, and idle timeout duration in seconds.",
			Value:       httputils.DefaultTimeoutSeconds,
			Destination: &app.serverTimeoutSeconds,
		},
		cli.IntFlag{
			Name:        "max-request-size",
			Usage:       "Maximum number of bytes the server will read parsing the request content.",
			Value:       httputils.DefaultMaxRequestBytes,
			Destination: &app.serverMaxRequestBytes,
		},
	}
}

const defaultModelFile = "spago_model.bin"

func newServerCommandActionFor(app *BartApp) func(c *cli.Context) error {
	return func(c *cli.Context) error {
		modelPath := filepath.Join(app.repo, app.model)

		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			fmt.Printf("Unable to find `%s` locally.\n", modelPath)
			fmt.Printf("Pulling `%s` from Hugging Face models hub...\n", app.model)
			// make sure the models path exists
			if _, err := os.Stat(app.repo); os.IsNotExist(err) {
				if err := os.MkdirAll(app.repo, 0755); err != nil {
					return err
				}
			}
			err = huggingface.NewDownloader(app.repo, app.model, false).Download()
			if err != nil {
				return err
			}
			fmt.Printf("Converting model...\n")
			err = huggingface.NewConverter(app.repo, app.model).Convert()
			if err != nil {
				return err
			}
		} else if _, err := os.Stat(path.Join(modelPath, defaultModelFile)); os.IsNotExist(err) {
			fmt.Printf("Unable to find `%s` in the model directory.\n", defaultModelFile)
			fmt.Printf("Assuming there is a Hugging Face model to convert...\n")
			err = huggingface.NewConverter(app.repo, app.model).Convert()
			if err != nil {
				return err
			}
		}

		tokenizer, err := bpetokenizer.NewFromModelFolder(modelPath)
		if err != nil {
			log.Fatal(err)
		}
		if tokenizer == nil {
			log.Fatal("expected BPETokenizer, actual nil")
		}

		model, err := barthead.LoadModelForSequenceClassification(modelPath)
		if err != nil {
			log.Fatal(err)
		}
		defer model.Close()
		fmt.Printf("Config: %+v\n", model.BART.Config)

		if !app.tlsDisable {
			fmt.Printf("TLS Cert path is %s\n", app.tlsCert)
			fmt.Printf("TLS private key path is %s\n", app.tlsKey)
		}

		fmt.Printf("Start %s gRPC server listening on %s.\n", func() string {
			if app.tlsDisable {
				return "non-TLS"
			}
			return "TLS"
		}(), app.grpcAddress)

		fmt.Printf("Start %s HTTP server listening on %s.\n", func() string {
			if app.tlsDisable {
				return "non-TLS"
			}
			return "TLS"
		}(), app.address)

		server := bartserver.NewServer(model, tokenizer)
		server.TimeoutSeconds = app.serverTimeoutSeconds
		server.MaxRequestBytes = app.serverMaxRequestBytes
		server.StartDefaultHTTPServer(app.address, app.tlsCert, app.tlsKey, app.tlsDisable)
		server.StartDefaultServer(app.grpcAddress, app.tlsCert, app.tlsKey, app.tlsDisable)

		return nil
	}
}
