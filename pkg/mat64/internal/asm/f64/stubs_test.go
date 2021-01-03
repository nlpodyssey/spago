// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64

import "testing"

func TestL1Norm(t *testing.T) {
	var srcGd float64 = 1
	for j, v := range []struct {
		want float64
		x    []float64
	}{
		{want: 0, x: []float64{}},
		{want: 2, x: []float64{2}},
		{want: 6, x: []float64{1, 2, 3}},
		{want: 6, x: []float64{-1, -2, -3}},
		{want: nan, x: []float64{nan}},
		{want: 40, x: []float64{8, -8, 8, -8, 8}},
		{want: 5, x: []float64{0, 1, 0, -1, 0, 1, 0, -1, 0, 1}},
	} {
		gLn := 4 + j%2
		v.x = guardVector(v.x, srcGd, gLn)
		src := v.x[gLn : len(v.x)-gLn]
		ret := L1Norm(src)
		if !same(ret, v.want) {
			t.Errorf("Test %d L1Norm error Got: %f Expected: %f", j, ret, v.want)
		}
		if !isValidGuard(v.x, srcGd, gLn) {
			t.Errorf("Test %d Guard violated in src vector %v %v", j, v.x[:gLn], v.x[len(v.x)-gLn:])
		}
	}
}

func TestL1NormInc(t *testing.T) {
	var srcGd float64 = 1
	for j, v := range []struct {
		inc  int
		want float64
		x    []float64
	}{
		{inc: 2, want: 0, x: []float64{}},
		{inc: 3, want: 2, x: []float64{2}},
		{inc: 10, want: 6, x: []float64{1, 2, 3}},
		{inc: 5, want: 6, x: []float64{-1, -2, -3}},
		{inc: 3, want: nan, x: []float64{nan}},
		{inc: 15, want: 40, x: []float64{8, -8, 8, -8, 8}},
		{inc: 1, want: 5, x: []float64{0, 1, 0, -1, 0, 1, 0, -1, 0, 1}},
	} {
		gLn, ln := 4+j%2, len(v.x)
		v.x = guardIncVector(v.x, srcGd, v.inc, gLn)
		src := v.x[gLn : len(v.x)-gLn]
		ret := L1NormInc(src, ln, v.inc)
		if !same(ret, v.want) {
			t.Errorf("Test %d L1NormInc error Got: %f Expected: %f", j, ret, v.want)
		}
		checkValidIncGuard(t, v.x, srcGd, v.inc, gLn)
	}
}

func TestAdd(t *testing.T) {
	var srcGd, dstGd float64 = 1, 0
	for j, v := range []struct {
		dst, src, expect []float64
	}{
		{
			dst:    []float64{1},
			src:    []float64{0},
			expect: []float64{1},
		},
		{
			dst:    []float64{1, 2, 3},
			src:    []float64{1},
			expect: []float64{2, 2, 3},
		},
		{
			dst:    []float64{},
			src:    []float64{},
			expect: []float64{},
		},
		{
			dst:    []float64{1},
			src:    []float64{nan},
			expect: []float64{nan},
		},
		{
			dst:    []float64{8, 8, 8, 8, 8},
			src:    []float64{2, 4, nan, 8, 9},
			expect: []float64{10, 12, nan, 16, 17},
		},
		{
			dst:    []float64{0, 1, 2, 3, 4},
			src:    []float64{-inf, 4, nan, 8, 9},
			expect: []float64{-inf, 5, nan, 11, 13},
		},
		{
			dst:    make([]float64, 50)[1:49],
			src:    make([]float64, 50)[1:49],
			expect: make([]float64, 50)[1:49],
		},
	} {
		sgLn, dgLn := 4+j%2, 4+j%3
		v.src, v.dst = guardVector(v.src, srcGd, sgLn), guardVector(v.dst, dstGd, dgLn)
		src, dst := v.src[sgLn:len(v.src)-sgLn], v.dst[dgLn:len(v.dst)-dgLn]
		Add(dst, src)
		for i := range v.expect {
			if !same(dst[i], v.expect[i]) {
				t.Errorf("Test %d Add error at %d Got: %v Expected: %v", j, i, dst[i], v.expect[i])
			}
		}
		if !isValidGuard(v.src, srcGd, sgLn) {
			t.Errorf("Test %d Guard violated in src vector %v %v", j, v.src[:sgLn], v.src[len(v.src)-sgLn:])
		}
		if !isValidGuard(v.dst, dstGd, dgLn) {
			t.Errorf("Test %d Guard violated in dst vector %v %v", j, v.dst[:dgLn], v.dst[len(v.dst)-dgLn:])
		}
	}
}

func TestAddConst(t *testing.T) {
	var srcGd float64 = 0
	for j, v := range []struct {
		alpha       float64
		src, expect []float64
	}{
		{
			alpha:  1,
			src:    []float64{0},
			expect: []float64{1},
		},
		{
			alpha:  5,
			src:    []float64{},
			expect: []float64{},
		},
		{
			alpha:  1,
			src:    []float64{nan},
			expect: []float64{nan},
		},
		{
			alpha:  8,
			src:    []float64{2, 4, nan, 8, 9},
			expect: []float64{10, 12, nan, 16, 17},
		},
		{
			alpha:  inf,
			src:    []float64{-inf, 4, nan, 8, 9},
			expect: []float64{nan, inf, nan, inf, inf},
		},
	} {
		gLn := 4 + j%2
		v.src = guardVector(v.src, srcGd, gLn)
		src := v.src[gLn : len(v.src)-gLn]
		AddConst(v.alpha, src)
		for i := range v.expect {
			if !same(src[i], v.expect[i]) {
				t.Errorf("Test %d AddConst error at %d Got: %v Expected: %v", j, i, src[i], v.expect[i])
			}
		}
		if !isValidGuard(v.src, srcGd, gLn) {
			t.Errorf("Test %d Guard violated in src vector %v %v", j, v.src[:gLn], v.src[len(v.src)-gLn:])
		}
	}
}

func TestCumSum(t *testing.T) {
	var srcGd, dstGd float64 = -1, 0
	for j, v := range []struct {
		dst, src, expect []float64
	}{
		{
			dst:    []float64{},
			src:    []float64{},
			expect: []float64{},
		},
		{
			dst:    []float64{0},
			src:    []float64{1},
			expect: []float64{1},
		},
		{
			dst:    []float64{nan},
			src:    []float64{nan},
			expect: []float64{nan},
		},
		{
			dst:    []float64{0, 0, 0},
			src:    []float64{1, 2, 3},
			expect: []float64{1, 3, 6},
		},
		{
			dst:    []float64{0, 0, 0, 0},
			src:    []float64{1, 2, 3},
			expect: []float64{1, 3, 6},
		},
		{
			dst:    []float64{0, 0, 0, 0},
			src:    []float64{1, 2, 3, 4},
			expect: []float64{1, 3, 6, 10},
		},
		{
			dst:    []float64{1, nan, nan, 1, 1},
			src:    []float64{1, 1, nan, 1, 1},
			expect: []float64{1, 2, nan, nan, nan},
		},
		{
			dst:    []float64{nan, 4, inf, -inf, 9},
			src:    []float64{inf, 4, nan, -inf, 9},
			expect: []float64{inf, inf, nan, nan, nan},
		},
		{
			dst:    make([]float64, 16),
			src:    []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			expect: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		},
	} {
		gLn := 4 + j%2
		v.src, v.dst = guardVector(v.src, srcGd, gLn), guardVector(v.dst, dstGd, gLn)
		src, dst := v.src[gLn:len(v.src)-gLn], v.dst[gLn:len(v.dst)-gLn]
		ret := CumSum(dst, src)
		for i := range v.expect {
			if !same(ret[i], v.expect[i]) {
				t.Errorf("Test %d CumSum error at %d Got: %v Expected: %v", j, i, ret[i], v.expect[i])
			}
			if !same(ret[i], dst[i]) {
				t.Errorf("Test %d CumSum ret/dst mismatch %d Ret: %v Dst: %v", j, i, ret[i], dst[i])
			}
		}
		if !isValidGuard(v.src, srcGd, gLn) {
			t.Errorf("Test %d Guard violated in src vector %v %v", j, v.src[:gLn], v.src[len(v.src)-gLn:])
		}
		if !isValidGuard(v.dst, dstGd, gLn) {
			t.Errorf("Test %d Guard violated in dst vector %v %v", j, v.dst[:gLn], v.dst[len(v.dst)-gLn:])
		}
	}
}

func TestCumProd(t *testing.T) {
	var srcGd, dstGd float64 = -1, 1
	for j, v := range []struct {
		dst, src, expect []float64
	}{
		{
			dst:    []float64{},
			src:    []float64{},
			expect: []float64{},
		},
		{
			dst:    []float64{1},
			src:    []float64{1},
			expect: []float64{1},
		},
		{
			dst:    []float64{nan},
			src:    []float64{nan},
			expect: []float64{nan},
		},
		{
			dst:    []float64{0, 0, 0, 0},
			src:    []float64{1, 2, 3, 4},
			expect: []float64{1, 2, 6, 24},
		},
		{
			dst:    []float64{0, 0, 0},
			src:    []float64{1, 2, 3},
			expect: []float64{1, 2, 6},
		},
		{
			dst:    []float64{0, 0, 0, 0},
			src:    []float64{1, 2, 3},
			expect: []float64{1, 2, 6},
		},
		{
			dst:    []float64{nan, 1, nan, 1, 0},
			src:    []float64{1, 1, nan, 1, 1},
			expect: []float64{1, 1, nan, nan, nan},
		},
		{
			dst:    []float64{nan, 4, nan, -inf, 9},
			src:    []float64{inf, 4, nan, -inf, 9},
			expect: []float64{inf, inf, nan, nan, nan},
		},
		{
			dst:    make([]float64, 18),
			src:    []float64{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
			expect: []float64{2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536},
		},
	} {
		sgLn, dgLn := 4+j%2, 4+j%3
		v.src, v.dst = guardVector(v.src, srcGd, sgLn), guardVector(v.dst, dstGd, dgLn)
		src, dst := v.src[sgLn:len(v.src)-sgLn], v.dst[dgLn:len(v.dst)-dgLn]
		ret := CumProd(dst, src)
		for i := range v.expect {
			if !same(ret[i], v.expect[i]) {
				t.Errorf("Test %d CumProd error at %d Got: %v Expected: %v", j, i, ret[i], v.expect[i])
			}
			if !same(ret[i], dst[i]) {
				t.Errorf("Test %d CumProd ret/dst mismatch %d Ret: %v Dst: %v", j, i, ret[i], dst[i])
			}
		}
		if !isValidGuard(v.src, srcGd, sgLn) {
			t.Errorf("Test %d Guard violated in src vector %v %v", j, v.src[:sgLn], v.src[len(v.src)-sgLn:])
		}
		if !isValidGuard(v.dst, dstGd, dgLn) {
			t.Errorf("Test %d Guard violated in dst vector %v %v", j, v.dst[:dgLn], v.dst[len(v.dst)-dgLn:])
		}
	}
}

func TestDiv(t *testing.T) {
	var srcGd, dstGd float64 = -1, 0.5
	for j, v := range []struct {
		dst, src, expect []float64
	}{
		{
			dst:    []float64{1},
			src:    []float64{1},
			expect: []float64{1},
		},
		{
			dst:    []float64{nan},
			src:    []float64{nan},
			expect: []float64{nan},
		},
		{
			dst:    []float64{1, 2, 3, 4},
			src:    []float64{1, 2, 3, 4},
			expect: []float64{1, 1, 1, 1},
		},
		{
			dst:    []float64{1, 2, 3, 4, 2, 4, 6, 8},
			src:    []float64{1, 2, 3, 4, 1, 2, 3, 4},
			expect: []float64{1, 1, 1, 1, 2, 2, 2, 2},
		},
		{
			dst:    []float64{2, 4, 6},
			src:    []float64{1, 2, 3},
			expect: []float64{2, 2, 2},
		},
		{
			dst:    []float64{0, 0, 0, 0},
			src:    []float64{1, 2, 3},
			expect: []float64{0, 0, 0},
		},
		{
			dst:    []float64{nan, 1, nan, 1, 0, nan, 1, nan, 1, 0},
			src:    []float64{1, 1, nan, 1, 1, 1, 1, nan, 1, 1},
			expect: []float64{nan, 1, nan, 1, 0, nan, 1, nan, 1, 0},
		},
		{
			dst:    []float64{inf, 4, nan, -inf, 9, inf, 4, nan, -inf, 9},
			src:    []float64{inf, 4, nan, -inf, 3, inf, 4, nan, -inf, 3},
			expect: []float64{nan, 1, nan, nan, 3, nan, 1, nan, nan, 3},
		},
	} {
		sgLn, dgLn := 4+j%2, 4+j%3
		v.src, v.dst = guardVector(v.src, srcGd, sgLn), guardVector(v.dst, dstGd, dgLn)
		src, dst := v.src[sgLn:len(v.src)-sgLn], v.dst[dgLn:len(v.dst)-dgLn]
		Div(dst, src)
		for i := range v.expect {
			if !same(dst[i], v.expect[i]) {
				t.Errorf("Test %d Div error at %d Got: %v Expected: %v", j, i, dst[i], v.expect[i])
			}
		}
		if !isValidGuard(v.src, srcGd, sgLn) {
			t.Errorf("Test %d Guard violated in src vector %v %v", j, v.src[:sgLn], v.src[len(v.src)-sgLn:])
		}
		if !isValidGuard(v.dst, dstGd, dgLn) {
			t.Errorf("Test %d Guard violated in dst vector %v %v", j, v.dst[:dgLn], v.dst[len(v.dst)-dgLn:])
		}
	}
}

func TestDivTo(t *testing.T) {
	var dstGd, xGd, yGd float64 = -1, 0.5, 0.25
	for j, v := range []struct {
		dst, x, y, expect []float64
	}{
		{
			dst:    []float64{1},
			x:      []float64{1},
			y:      []float64{1},
			expect: []float64{1},
		},
		{
			dst:    []float64{1},
			x:      []float64{nan},
			y:      []float64{nan},
			expect: []float64{nan},
		},
		{
			dst:    []float64{-2, -2, -2},
			x:      []float64{1, 2, 3},
			y:      []float64{1, 2, 3},
			expect: []float64{1, 1, 1},
		},
		{
			dst:    []float64{0, 0, 0},
			x:      []float64{2, 4, 6},
			y:      []float64{1, 2, 3, 4},
			expect: []float64{2, 2, 2},
		},
		{
			dst:    []float64{-1, -1, -1},
			x:      []float64{0, 0, 0},
			y:      []float64{1, 2, 3},
			expect: []float64{0, 0, 0},
		},
		{
			dst:    []float64{inf, inf, inf, inf, inf, inf, inf, inf, inf, inf},
			x:      []float64{nan, 1, nan, 1, 0, nan, 1, nan, 1, 0},
			y:      []float64{1, 1, nan, 1, 1, 1, 1, nan, 1, 1},
			expect: []float64{nan, 1, nan, 1, 0, nan, 1, nan, 1, 0},
		},
		{
			dst:    []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			x:      []float64{inf, 4, nan, -inf, 9, inf, 4, nan, -inf, 9},
			y:      []float64{inf, 4, nan, -inf, 3, inf, 4, nan, -inf, 3},
			expect: []float64{nan, 1, nan, nan, 3, nan, 1, nan, nan, 3},
		},
	} {
		xgLn, ygLn := 4+j%2, 4+j%3
		v.y, v.x = guardVector(v.y, yGd, ygLn), guardVector(v.x, xGd, xgLn)
		y, x := v.y[ygLn:len(v.y)-ygLn], v.x[xgLn:len(v.x)-xgLn]
		v.dst = guardVector(v.dst, dstGd, xgLn)
		dst := v.dst[xgLn : len(v.dst)-xgLn]
		ret := DivTo(dst, x, y)
		for i := range v.expect {
			if !same(ret[i], v.expect[i]) {
				t.Errorf("Test %d DivTo error at %d Got: %v Expected: %v", j, i, ret[i], v.expect[i])
			}
			if !same(ret[i], dst[i]) {
				t.Errorf("Test %d DivTo ret/dst mismatch %d Ret: %v Dst: %v", j, i, ret[i], dst[i])
			}
		}
		if !isValidGuard(v.y, yGd, ygLn) {
			t.Errorf("Test %d Guard violated in y vector %v %v", j, v.y[:ygLn], v.y[len(v.y)-ygLn:])
		}
		if !isValidGuard(v.x, xGd, xgLn) {
			t.Errorf("Test %d Guard violated in x vector %v %v", j, v.x[:xgLn], v.x[len(v.x)-xgLn:])
		}
		if !isValidGuard(v.dst, dstGd, xgLn) {
			t.Errorf("Test %d Guard violated in dst vector %v %v", j, v.dst[:xgLn], v.dst[len(v.dst)-xgLn:])
		}
	}
}

func TestL1Dist(t *testing.T) {
	var tGd, sGd = -inf, inf
	for j, v := range []struct {
		s, t   []float64
		expect float64
	}{
		{
			s:      []float64{1},
			t:      []float64{1},
			expect: 0,
		},
		{
			s:      []float64{nan},
			t:      []float64{nan},
			expect: nan,
		},
		{
			s:      []float64{1, 2, 3, 4},
			t:      []float64{1, 2, 3, 4},
			expect: 0,
		},
		{
			s:      []float64{2, 4, 6},
			t:      []float64{1, 2, 3, 4},
			expect: 6,
		},
		{
			s:      []float64{0, 0, 0},
			t:      []float64{1, 2, 3},
			expect: 6,
		},
		{
			s:      []float64{0, -4, -10},
			t:      []float64{1, 2, 3},
			expect: 20,
		},
		{
			s:      []float64{0, 1, 0, 1, 0},
			t:      []float64{1, 1, inf, 1, 1},
			expect: inf,
		},
		{
			s:      []float64{inf, 4, nan, -inf, 9},
			t:      []float64{inf, 4, nan, -inf, 3},
			expect: nan,
		},
	} {
		sgLn, tgLn := 4+j%2, 4+j%3
		v.s, v.t = guardVector(v.s, sGd, sgLn), guardVector(v.t, tGd, tgLn)
		sLc, tLc := v.s[sgLn:len(v.s)-sgLn], v.t[tgLn:len(v.t)-tgLn]
		ret := L1Dist(sLc, tLc)
		if !same(ret, v.expect) {
			t.Errorf("Test %d L1Dist error Got: %f Expected: %f", j, ret, v.expect)
		}
		if !isValidGuard(v.s, sGd, sgLn) {
			t.Errorf("Test %d Guard violated in s vector %v %v", j, v.s[:sgLn], v.s[len(v.s)-sgLn:])
		}
		if !isValidGuard(v.t, tGd, tgLn) {
			t.Errorf("Test %d Guard violated in t vector %v %v", j, v.t[:tgLn], v.t[len(v.t)-tgLn:])
		}
	}
}

func TestLinfDist(t *testing.T) {
	var tGd, sGd float64 = 0, inf
	for j, v := range []struct {
		s, t   []float64
		expect float64
	}{
		{
			s:      []float64{},
			t:      []float64{},
			expect: 0,
		},
		{
			s:      []float64{1},
			t:      []float64{1},
			expect: 0,
		},
		{
			s:      []float64{nan},
			t:      []float64{nan},
			expect: nan,
		},
		{
			s:      []float64{1, 2, 3, 4},
			t:      []float64{1, 2, 3, 4},
			expect: 0,
		},
		{
			s:      []float64{2, 4, 6},
			t:      []float64{1, 2, 3, 4},
			expect: 3,
		},
		{
			s:      []float64{0, 0, 0},
			t:      []float64{1, 2, 3},
			expect: 3,
		},
		{
			s:      []float64{0, 1, 0, 1, 0},
			t:      []float64{1, 1, inf, 1, 1},
			expect: inf,
		},
		{
			s:      []float64{inf, 4, nan, -inf, 9},
			t:      []float64{inf, 4, nan, -inf, 3},
			expect: 6,
		},
	} {
		sgLn, tgLn := 4+j%2, 4+j%3
		v.s, v.t = guardVector(v.s, sGd, sgLn), guardVector(v.t, tGd, tgLn)
		sLc, tLc := v.s[sgLn:len(v.s)-sgLn], v.t[tgLn:len(v.t)-tgLn]
		ret := LinfDist(sLc, tLc)
		if !same(ret, v.expect) {
			t.Errorf("Test %d LinfDist error Got: %f Expected: %f", j, ret, v.expect)
		}
		if !isValidGuard(v.s, sGd, sgLn) {
			t.Errorf("Test %d Guard violated in s vector %v %v", j, v.s[:sgLn], v.s[len(v.s)-sgLn:])
		}
		if !isValidGuard(v.t, tGd, tgLn) {
			t.Errorf("Test %d Guard violated in t vector %v %v", j, v.t[:tgLn], v.t[len(v.t)-tgLn:])
		}
	}
}

func TestSum(t *testing.T) {
	var srcGd float64 = -1
	for j, v := range []struct {
		src    []float64
		expect float64
	}{
		{
			src:    []float64{},
			expect: 0,
		},
		{
			src:    []float64{1},
			expect: 1,
		},
		{
			src:    []float64{nan},
			expect: nan,
		},
		{
			src:    []float64{1, 2, 3},
			expect: 6,
		},
		{
			src:    []float64{1, -4, 3},
			expect: 0,
		},
		{
			src:    []float64{1, 2, 3, 4},
			expect: 10,
		},
		{
			src:    []float64{1, 1, nan, 1, 1},
			expect: nan,
		},
		{
			src:    []float64{inf, 4, nan, -inf, 9},
			expect: nan,
		},
		{
			src:    []float64{1, 1, 1, 1, 9, 1, 1, 1, 2, 1, 1, 1, 1, 1, 5, 1},
			expect: 29,
		},
		{
			src:    []float64{1, 1, 1, 1, 9, 1, 1, 1, 2, 1, 1, 1, 1, 1, 5, 11, 1, 1, 1, 9, 1, 1, 1, 2, 1, 1, 1, 1, 1, 5, 1},
			expect: 67,
		},
	} {
		gdLn := 4 + j%2
		gsrc := guardVector(v.src, srcGd, gdLn)
		src := gsrc[gdLn : len(gsrc)-gdLn]
		ret := Sum(src)
		if !same(ret, v.expect) {
			t.Errorf("Test %d Sum error Got: %v Expected: %v", j, ret, v.expect)
		}
		if !isValidGuard(gsrc, srcGd, gdLn) {
			t.Errorf("Test %d Guard violated in src vector %v %v", j, gsrc[:gdLn], gsrc[len(gsrc)-gdLn:])
		}
	}
}
