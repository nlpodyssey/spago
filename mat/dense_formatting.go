// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"bytes"
	"fmt"
	"strconv"

	"github.com/nlpodyssey/spago/mat/float"
)

// Format implements custom formatting for representing a Dense matrix.
// Thanks to this method, a Dense matrix satisfies the fmt.Formatter interface.
func (d *Dense[_]) Format(f fmt.State, c rune) {
	if c == 'v' {
		if f.Flag('#') {
			fmt.Fprintf(f, "&%#v", d.data) // TODO: print the shape
			return
		}
		if f.Flag('+') {
			fmt.Fprintf(f, "%+v", d.data) // TODO: print the shape
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
func (d *Dense[_]) format(f fmt.State, c rune, precision int) {
	maxWidths, maxWidth := d.formattingMaxColumnsWidth(f, c, precision)
	spaceBuf := makeSpaceBuffer(maxWidth)
	buf := make([]byte, 0, maxWidth)
	for row, index := 0, 0; row < d.shape[0]; row++ {
		rowPrefix, rowSuffix := d.formattingRowPrefixAndSuffix(row)
		fmt.Fprint(f, rowPrefix)

		for col := 0; col < d.shape[1]; col, index = col+1, index+1 {
			if col > 0 {
				fmt.Fprint(f, " ")
			}
			buf = formatValue(buf, d.data[index], c, precision)
			writeFormattedValue(f, buf, spaceBuf, maxWidths[col])
		}

		fmt.Fprint(f, rowSuffix)
	}
}

func writeFormattedValue(f fmt.State, buf, spaceBuf []byte, maxW lrWidth) {
	var leftPadding, rightPadding int

	if pi := bytes.IndexByte(buf, '.'); pi != -1 {
		leftPadding = maxW.left - pi
		rightPadding = maxW.right - (len(buf) - pi - 1)
	} else {
		leftPadding = maxW.left - len(buf)
		rightPadding = maxW.right
		if rightPadding > 0 {
			rightPadding++
		}
	}

	f.Write(spaceBuf[:leftPadding])
	f.Write(buf)
	f.Write(spaceBuf[:rightPadding])
}

func makeSpaceBuffer(length int) []byte {
	buf := make([]byte, length)
	for i := range buf {
		buf[i] = ' '
	}
	return buf
}

func (d *Dense[_]) formattingRowPrefixAndSuffix(rowIndex int) (string, string) {
	if d.shape[0] == 1 {
		return "[", "]"
	}
	if rowIndex == 0 {
		return "⎡", "⎤\n"
	}
	if rowIndex == d.shape[0]-1 {
		return "⎣", "⎦"
	}
	return "⎢", "⎥\n"
}

type lrWidth struct{ left, right int }

func (d *Dense[_]) formattingMaxColumnsWidth(
	f fmt.State,
	c rune,
	precision int,
) ([]lrWidth, int) {
	minWidth := 0
	if fw, ok := f.Width(); ok {
		minWidth = fw
	}

	maxWidth := 0
	widths := make([]lrWidth, d.shape[1])

	buf := make([]byte, 0, 16)
	for row, index := 0, 0; row < d.shape[0]; row++ {
		for col := 0; col < d.shape[1]; col, index = col+1, index+1 {
			buf = formatValue(buf, d.data[index], c, precision)
			w := len(buf)
			maxWidth = maxInt(maxWidth, w)
			if pi := bytes.IndexByte(buf, '.'); pi != -1 {
				leftSize := pi
				if minWidth > w {
					leftSize += minWidth - w
				}
				widths[col].left = maxInt(widths[col].left, leftSize)
				widths[col].right = maxInt(widths[col].right, w-pi-1)
			} else {
				widths[col].left = maxInt(widths[col].left, w)
			}
		}
	}

	return widths, maxWidth
}

func formatValue[T float.DType](buf []byte, val T, c rune, precision int) []byte {
	return strconv.AppendFloat(buf[:0], float64(val), byte(c), precision, 32)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
