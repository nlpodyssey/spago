// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"strconv"

	_ "github.com/nlpodyssey/spago/pkg/global"
)

var _ fmt.Formatter = &Dense{}

// Format implements custom formatting for represeinting a Dense matrix.
// Thanks to this method, a Dense matrix implements the fmt.Formatter interface.
func (d *Dense) Format(f fmt.State, c rune) {
	if c == 'v' {
		if f.Flag('#') {
			fmt.Fprintf(f, "&%#v", *d)
			return
		}
		if f.Flag('+') {
			fmt.Fprintf(f, "%+v", *d)
			return
		}
		c = 'g'
	}

	if len(d.data) == 0 {
		fmt.Fprintf(f, "[]")
		return
	}

	if c == 'F' {
		c = 'f' // %F (alias for %f) does not work with strconv.AppendFloat
	}

	precision, precisionOk := f.Precision()
	if !precisionOk {
		precision = -1
	}

	d.format(f, c, precision)
}

// format formats a non-empty Dense matrix
func (d *Dense) format(f fmt.State, c rune, precision int) {
	columnWidths, maxWidth := d.formattingMaxColumnsWidth(f, c, precision)

	spaceBuf := make([]byte, maxWidth+1)
	for i := range spaceBuf {
		spaceBuf[i] = ' '
	}

	buf := make([]byte, 0)
	for row, index := 0, 0; row < d.rows; row++ {
		rowPrefix, rowSuffix := d.formattingRowPrefixAndSuffix(row)
		fmt.Fprintf(f, rowPrefix)

		for col := 0; col < d.cols; col, index = col+1, index+1 {
			buf = strconv.AppendFloat(buf[:0], d.data[index], byte(c), precision, 64)
			paddingSize := columnWidths[col] - len(buf)
			if col > 0 {
				paddingSize++
			}
			f.Write(spaceBuf[:paddingSize])
			f.Write(buf)
		}

		fmt.Fprintf(f, rowSuffix)
	}
}

func (d *Dense) formattingRowPrefixAndSuffix(rowIndex int) (string, string) {
	if d.rows == 1 {
		return "[", "]"
	}
	if rowIndex == 0 {
		return "⎡", "⎤\n"
	}
	if rowIndex == d.rows-1 {
		return "⎣", "⎦"
	}
	return "⎢", "⎥\n"
}

func (d *Dense) formattingMaxColumnsWidth(
	f fmt.State,
	c rune,
	precision int,
) ([]int, int) {
	minWidth := 0
	if fw, ok := f.Width(); ok {
		minWidth = fw
	}

	widths := make([]int, d.cols)
	maxWidth := 0

	buf := make([]byte, 0)
	for row, index := 0, 0; row < d.rows; row++ {
		for col := 0; col < d.cols; col, index = col+1, index+1 {
			buf = strconv.AppendFloat(buf[:0], d.data[index], byte(c), precision, 64)
			w := maxInt(len(buf), minWidth)
			widths[col] = maxInt(widths[col], w)
			maxWidth = maxInt(maxWidth, w)
		}
	}

	return widths, maxWidth
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
