// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import "strings"

// BeforeSpace returns the substring from value which comes before the first whitespace.
func BeforeSpace(value string) string {
	return Before(value, " ")
}

// AfterSpace returns the substring from value which comes after the first whitespace.
func AfterSpace(value string) string {
	return After(value, " ")
}

// Before returns the substring from value which comes before a.
func Before(value string, a string) string {
	pos := strings.Index(value, a)
	if pos == -1 {
		return ""
	}
	return value[0:pos]
}

// After returns the substring from value which comes after a.
func After(value string, a string) string {
	pos := strings.Index(value, a)
	if pos == -1 {
		return ""
	}
	adjustedPos := pos + len(a)
	if adjustedPos >= len(value) {
		return ""
	}
	return value[adjustedPos:]
}

// SplitByRune splits the input string by runes, and produces a new slice
// of string, where each item is one rune converted to string.
func SplitByRune(str string) []string {
	out := make([]string, 0)
	for _, item := range str {
		out = append(out, string(item))
	}
	return out
}

// ReverseString reverses the given string.
func ReverseString(text string) string {
	if len(text) == 0 {
		return ""
	}
	str := []rune(text)
	l := len(str)
	revStr := make([]rune, l)
	for i := 0; i <= l/2; i++ {
		revStr[i], revStr[l-1-i] = str[l-1-i], str[i]
	}
	return string(revStr)
}
