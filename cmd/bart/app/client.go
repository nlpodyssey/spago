// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"github.com/urfave/cli"
)

func newClientCommandFor(app *BartApp) cli.Command {
	return cli.Command{
		Name:  "client",
		Usage: "Run the " + programName + " client.",
		Subcommands: []cli.Command{
			newClientClassifyCommandFor(app),
			newClientClassifyNLICommandFor(app),
		},
	}
}
