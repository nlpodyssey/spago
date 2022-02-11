// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"context"
	"github.com/nlpodyssey/spago/cmd/clientutils"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"github.com/urfave/cli/v2"
	"math"
)

func newClientSimilarityCommandFor[T mat.DType](app *BertApp) *cli.Command {
	return &cli.Command{
		Name:        "similarity",
		Usage:       "Perform text-similarity using BERT sentence encoding.",
		Description: "Run the " + programName + " client to determine the similarity between two texts.",
		Flags:       newClientSimilarityCommandFlagsFor(app),
		Action:      newClientSimilarityCommandActionFor[T](app),
	}
}

func newClientSimilarityCommandFlagsFor(app *BertApp) []cli.Flag {
	return clientutils.Flags(&app.address, &app.tlsDisable, &app.output, []cli.Flag{
		&cli.StringFlag{
			Name:        "text1",
			Destination: &app.requestText,
			Required:    true,
		},
		&cli.StringFlag{
			Name:        "text2",
			Destination: &app.requestText2,
			Required:    true,
		},
	})
}

func newClientSimilarityCommandActionFor[T mat.DType](app *BertApp) func(c *cli.Context) error {
	return func(c *cli.Context) error {
		clientutils.VerifyFlags(app.output)

		conn := clientutils.OpenConnection(app.address, app.tlsDisable)
		client := grpcapi.NewBERTClient(conn)

		resp, err := client.Encode(context.Background(), &grpcapi.EncodeRequest{
			Text: app.requestText,
		})
		if err != nil {
			return err
		}

		resp2, err := client.Encode(context.Background(), &grpcapi.EncodeRequest{
			Text: app.requestText2,
		})
		if err != nil {
			return err
		}

		vec1 := normalize(f32SliceToFloatSlice[T](resp.Vector))
		vec2 := normalize(f32SliceToFloatSlice[T](resp2.Vector))
		similarity := vec1.DotUnitary(vec2)

		clientutils.Println(app.output, toFixed(similarity, 6))

		return nil
	}
}

func normalize[T mat.DType](xs []T) mat.Matrix[T] {
	return mat.NewVecDense(xs).Normalize2()
}

func f32SliceToFloatSlice[T mat.DType](xs []float32) []T {
	ys := make([]T, len(xs))
	for i, f32 := range xs {
		ys[i] = T(f32)
	}
	return ys
}

func round[T mat.DType](num T) int {
	return int(float64(num) + math.Copysign(0.5, float64(num)))
}

func toFixed[T mat.DType](num T, precision int) T {
	output := mat.Pow(10, T(precision))
	return T(round(num*output)) / output
}
