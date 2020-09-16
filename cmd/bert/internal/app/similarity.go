// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"context"
	"github.com/nlpodyssey/spago/cmd/clientutils"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"github.com/urfave/cli"
	"log"
	"math"
)

func newClientSimilarityCommandFor(app *BertApp) cli.Command {
	return cli.Command{
		Name:        "similarity",
		Usage:       "Perform text-similarity using BERT sentence encoding.",
		UsageText:   programName + " client encode --text=<value>" + clientutils.UsageText(),
		Description: "Run the " + programName + " client to determine the similarity between two texts.",
		Flags:       newClientSimilarityCommandFlagsFor(app),
		Action:      newClientSimilarityCommandActionFor(app),
	}
}

func newClientSimilarityCommandFlagsFor(app *BertApp) []cli.Flag {
	return clientutils.Flags(&app.address, &app.tlsDisable, &app.output, []cli.Flag{
		cli.StringFlag{
			Name:        "text1",
			Destination: &app.requestText,
			Required:    true,
		},
		cli.StringFlag{
			Name:        "text2",
			Destination: &app.requestText2,
			Required:    true,
		},
	})
}

func newClientSimilarityCommandActionFor(app *BertApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		clientutils.VerifyFlags(app.output)

		conn := clientutils.OpenConnection(app.address, app.tlsDisable)
		client := grpcapi.NewBERTClient(conn)

		resp, err := client.Encode(context.Background(), &grpcapi.EncodeRequest{
			Text: app.requestText,
		})
		if err != nil {
			log.Fatalln(err)
		}

		resp2, err := client.Encode(context.Background(), &grpcapi.EncodeRequest{
			Text: app.requestText2,
		})
		if err != nil {
			log.Fatalln(err)
		}

		vec1 := mat.NewVecDense(f32tof64(resp.Vector))
		vec2 := mat.NewVecDense(f32tof64(resp2.Vector))
		similarity := vec1.DotUnitary(vec2)

		clientutils.Println(app.output, toFixed(similarity, 6))
	}
}

func f32tof64(xs []float32) []float64 {
	ys := make([]float64, len(xs))
	for i, f32 := range xs {
		ys[i] = float64(f32)
	}
	return ys
}

func round(num float64) int {
	return int(num + math.Copysign(0.5, num))
}

func toFixed(num float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return float64(round(num*output)) / output
}
