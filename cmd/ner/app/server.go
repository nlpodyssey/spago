// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"fmt"
	"log"
	"os"
	"os/user"
	"path"
	"path/filepath"

	"github.com/nlpodyssey/spago/pkg/nlp/sequencelabeler"
	"github.com/nlpodyssey/spago/pkg/utils/httputils"
	"github.com/urfave/cli"
)

func newServerCommandFor(app *NERApp) cli.Command {
	return cli.Command{
		Name:        "server",
		Usage:       "Run the " + programName + " as gRPC/HTTP server.",
		UsageText:   programName + " server --model=<name> [--repo=<path>] [--address=<address>] [--tls-cert-file=<cert>] [--tls-key-file=<key>] [--tls-disable]",
		Description: "You must indicate the directory that contains the spaGO neural models.",
		Flags:       newServerCommandFlagsFor(app),
		Action:      newServerCommandActionFor(app),
	}
}

func newServerCommandFlagsFor(app *NERApp) []cli.Flag {
	usr, err := user.Current()
	if err != nil {
		log.Fatal(err)
	}

	return []cli.Flag{
		cli.StringFlag{
			Name:        "address",
			Usage:       "Specifies the bind-address of the server.",
			Value:       "0.0.0.0:1987",
			Destination: &app.address,
		},
		cli.StringFlag{
			Name:        "grpc-address",
			Usage:       "Changes the bind address of the gRPC server.",
			Value:       "0.0.0.0:1976",
			Destination: &app.grpcAddress,
		},
		cli.StringFlag{
			Name:        "repo",
			Usage:       "Specifies the path to the models.",
			Value:       path.Join(usr.HomeDir, ".spago"),
			Destination: &app.repo,
		},
		cli.StringFlag{
			Name:        "model",
			Usage:       "Specifies the name of the model to use.",
			Destination: &app.modelName,
			Required:    true,
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
			Name:        "tls-disable ",
			Usage:       "Specifies that TLS is disabled.",
			Destination: &app.tlsDisable,
		},
	}
}

func newServerCommandActionFor(app *NERApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		fmt.Printf("TLS Cert path is %s\n", app.tlsCert)
		fmt.Printf("TLS private key path is %s\n", app.tlsKey)

		modelsFolder := app.repo
		if _, err := os.Stat(modelsFolder); os.IsNotExist(err) {
			log.Fatal(err)
		}

		modelName := app.modelName
		modelPath := filepath.Join(modelsFolder, modelName)
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			switch url, ok := predefinedModels[modelName]; {
			case ok:
				fmt.Printf("Fetch model from `%s`\n", url)
				if err := httputils.DownloadFile(fmt.Sprintf("%s-compressed", modelPath), url); err != nil {
					log.Fatal(err)
				}
				r, err := os.Open(fmt.Sprintf("%s-compressed", modelPath))
				if err != nil {
					log.Fatal(err)
				}
				fmt.Print("Extracting compressed model... ")
				extractTarGz(r, modelsFolder)
				fmt.Println("ok")
			default:
				log.Fatal(err)
			}
		}

		configPath := filepath.Join(modelPath, "config.json")
		config := sequencelabeler.LoadConfig(configPath)
		model := sequencelabeler.NewDefaultModel(config, modelPath, true, false)
		model.Load(modelPath)
		model.LoadEmbeddings(config, modelPath, true, false) // TODO: find a general solution

		fmt.Printf("Start %s HTTP server listening on %s.\n", func() string {
			if app.tlsDisable {
				return "non-TLS"
			}
			return "TLS"
		}(), app.address)

		fmt.Printf("Start %s gRPC server listening on %s.\n", func() string {
			if app.tlsDisable {
				return "non-TLS"
			}
			return "TLS"
		}(), app.grpcAddress)

		server := sequencelabeler.NewServer(model)
		server.Start(app.address, app.grpcAddress, app.tlsCert, app.tlsKey, app.tlsDisable)
	}
}
