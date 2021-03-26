// Copyright (c) 2020 Vikesh Raj C. All rights reserved.
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

package sentencepiece

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

func sanitize(s string) string {
	return norm.NFKC.String(s)
}

func normalize(s string) string {
	replacer := func(r rune) rune {
		if isControl(r) || r == 0 {
			return -1
		}
		if unicode.IsSpace(r) {
			return ' '
		}
		return r
	}
	return sanitize(strings.Map(replacer, s))
}

var controlChars = []rune{
	0x007F, 0x00AD, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605, 0x061C, 0x06DD, 0x070F,
	0x08E2, 0x180E, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D,
	0x202E, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0x2066, 0x2067, 0x2068, 0x2069, 0x206A,
	0x206B, 0x206C, 0x206D, 0x206E, 0x206F, 0xFEFF, 0xFFF9, 0xFFFA, 0xFFFB, 0x110BD,
	0x110CD, 0x13430, 0x13431, 0x13432, 0x13433, 0x13434, 0x13435, 0x13436, 0x13437,
	0x13438, 0x1BCA0, 0x1BCA1, 0x1BCA2, 0x1BCA3, 0x1D173, 0x1D174, 0x1D175, 0x1D176,
	0x1D177, 0x1D178, 0x1D179, 0x1D17A, 0xE0001,
}

//gocyclo:ignore
func isControl(c rune) bool {
	if c == ' ' || c == '\n' || c == '\r' || c == '\t' {
		return false
	}
	if c <= 0x001F || (c >= 0x0080 && c <= 0x009F) ||
		(c >= 0xE0020 && c <= 0xE007F) ||
		(c >= 0xE000 && c <= 0xF8FF) ||
		(c >= 0xF0000 && c <= 0xFFFFD) ||
		(c >= 0x100000 && c <= 0x10FFFD) ||
		(c >= 0xD800 && c <= 0xDB7F) ||
		(c >= 0xDB80 && c <= 0xDBFF) ||
		(c >= 0xDC00 && c <= 0xDFFF) ||
		isControlChar(c) {
		return true
	}
	return false
}

func isControlChar(c rune) bool {
	for _, ch := range controlChars {
		if ch == c {
			return true
		}
	}
	return false
}
