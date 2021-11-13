// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math32_test

import (
	"fmt"
	"testing"

	. "github.com/nlpodyssey/spago/pkg/mat32/internal/math32"
)

var vf = []float32{
	4.9790119248836735e+00,
	7.7388724745781045e+00,
	-2.7688005719200159e-01,
	-5.0106036182710749e+00,
	9.6362937071984173e+00,
	2.9263772392439646e+00,
	5.2290834314593066e+00,
	2.7279399104360102e+00,
	1.8253080916808550e+00,
	-8.6859247685756013e+00,
}

// The expected results below were computed by the high precision calculators
// at http://keisan.casio.com/.  More exact input values (array vf[], above)
// were obtained by printing them with "%.26f".  The answers were calculated
// to 26 digits (by using the "Digit number" drop-down control of each
// calculator).

var ceil = []float32{
	5.0000000000000000e+00,
	8.0000000000000000e+00,
	0.0000000000000000e+00,
	-5.0000000000000000e+00,
	1.0000000000000000e+01,
	3.0000000000000000e+00,
	6.0000000000000000e+00,
	3.0000000000000000e+00,
	2.0000000000000000e+00,
	-8.0000000000000000e+00,
}
var copysign = []float32{
	-4.9790119248836735e+00,
	-7.7388724745781045e+00,
	-2.7688005719200159e-01,
	-5.0106036182710749e+00,
	-9.6362937071984173e+00,
	-2.9263772392439646e+00,
	-5.2290834314593066e+00,
	-2.7279399104360102e+00,
	-1.8253080916808550e+00,
	-8.6859247685756013e+00,
}

var exp = []float32{
	1.4533071302642137507696589e+02,
	2.2958822575694449002537581e+03,
	7.5814542574851666582042306e-01,
	6.6668778421791005061482264e-03,
	1.5310493273896033740861206e+04,
	1.8659907517999328638667732e+01,
	1.8662167355098714543942057e+02,
	1.5301332413189378961665788e+01,
	6.2047063430646876349125085e+00,
	1.6894712385826521111610438e-04,
}
var fabs = []float32{
	4.9790119248836735e+00,
	7.7388724745781045e+00,
	2.7688005719200159e-01,
	5.0106036182710749e+00,
	9.6362937071984173e+00,
	2.9263772392439646e+00,
	5.2290834314593066e+00,
	2.7279399104360102e+00,
	1.8253080916808550e+00,
	8.6859247685756013e+00,
}
var fdim = []float32{
	4.9790119248836735e+00,
	7.7388724745781045e+00,
	0.0000000000000000e+00,
	0.0000000000000000e+00,
	9.6362937071984173e+00,
	2.9263772392439646e+00,
	5.2290834314593066e+00,
	2.7279399104360102e+00,
	1.8253080916808550e+00,
	0.0000000000000000e+00,
}
var floor = []float32{
	4.0000000000000000e+00,
	7.0000000000000000e+00,
	-1.0000000000000000e+00,
	-6.0000000000000000e+00,
	9.0000000000000000e+00,
	2.0000000000000000e+00,
	5.0000000000000000e+00,
	2.0000000000000000e+00,
	1.0000000000000000e+00,
	-9.0000000000000000e+00,
}

type fi struct {
	f float32
	i int
}

var frexp = []fi{
	{6.2237649061045918750e-01, 3},
	{9.6735905932226306250e-01, 3},
	{-5.5376011438400318000e-01, -1},
	{-6.2632545228388436250e-01, 3},
	{6.02268356699901081250e-01, 4},
	{7.3159430981099115000e-01, 2},
	{6.5363542893241332500e-01, 3},
	{6.8198497760900255000e-01, 2},
	{9.1265404584042750000e-01, 1},
	{-5.4287029803597508250e-01, 4},
}
var log = []float32{
	1.605231462693062999102599e+00,
	2.0462560018708770653153909e+00,
	-1.2841708730962657801275038e+00,
	1.6115563905281545116286206e+00,
	2.2655365644872016636317461e+00,
	1.0737652208918379856272735e+00,
	1.6542360106073546632707956e+00,
	1.0035467127723465801264487e+00,
	6.0174879014578057187016475e-01,
	2.161703872847352815363655e+00,
}
var modf = [][2]float32{
	{4.0000000000000000e+00, 9.7901192488367350108546816e-01},
	{7.0000000000000000e+00, 7.3887247457810456552351752e-01},
	{Copysign(0, -1), -2.7688005719200159404635997e-01},
	{-5.0000000000000000e+00, -1.060361827107492160848778e-02},
	{9.0000000000000000e+00, 6.3629370719841737980004837e-01},
	{2.0000000000000000e+00, 9.2637723924396464525443662e-01},
	{5.0000000000000000e+00, 2.2908343145930665230025625e-01},
	{2.0000000000000000e+00, 7.2793991043601025126008608e-01},
	{1.0000000000000000e+00, 8.2530809168085506044576505e-01},
	{-8.0000000000000000e+00, -6.8592476857560136238589621e-01},
}

var pow = []float32{
	9.5282232631648411840742957e+04,
	5.4811599352999901232411871e+07,
	5.2859121715894396531132279e-01,
	9.7587991957286474464259698e-06,
	4.328064329346044846740467e+09,
	8.4406761805034547437659092e+02,
	1.6946633276191194947742146e+05,
	5.3449040147551939075312879e+02,
	6.688182138451414936380374e+01,
	2.0609869004248742886827439e-09,
}
var signbit = []bool{
	false,
	false,
	true,
	true,
	false,
	false,
	false,
	false,
	false,
	true,
}

var sqrt = []float32{
	2.2313699659365484748756904e+00,
	2.7818829009464263511285458e+00,
	5.2619393496314796848143251e-01,
	2.2384377628763938724244104e+00,
	3.1042380236055381099288487e+00,
	1.7106657298385224403917771e+00,
	2.286718922705479046148059e+00,
	1.6516476350711159636222979e+00,
	1.3510396336454586262419247e+00,
	2.9471892997524949215723329e+00,
}

var tanh = []float32{
	9.9990531206936338549262119e-01,
	9.9999962057085294197613294e-01,
	-2.7001505097318677233756845e-01,
	-9.9991110943061718603541401e-01,
	9.9999999146798465745022007e-01,
	9.9427249436125236705001048e-01,
	9.9994257600983138572705076e-01,
	9.9149409509772875982054701e-01,
	9.4936501296239685514466577e-01,
	-9.9999994291374030946055701e-01,
}
var trunc = []float32{
	4.0000000000000000e+00,
	7.0000000000000000e+00,
	-0.0000000000000000e+00,
	-5.0000000000000000e+00,
	9.0000000000000000e+00,
	2.0000000000000000e+00,
	5.0000000000000000e+00,
	2.0000000000000000e+00,
	1.0000000000000000e+00,
	-8.0000000000000000e+00,
}

// arguments and expected results for special cases

var vfceilSC = []float32{
	Inf(-1),
	Copysign(0, -1),
	0,
	Inf(1),
	NaN(),
}
var ceilSC = []float32{
	Inf(-1),
	Copysign(0, -1),
	0,
	Inf(1),
	NaN(),
}

var vfcopysignSC = []float32{
	Inf(-1),
	Inf(1),
	NaN(),
}
var copysignSC = []float32{
	Inf(-1),
	Inf(-1),
	NaN(),
}

var vfexpSC = []float32{
	Inf(-1),
	-2000,
	2000,
	Inf(1),
	NaN(),
}
var expSC = []float32{
	0,
	0,
	Inf(1),
	Inf(1),
	NaN(),
}

var vffabsSC = []float32{
	Inf(-1),
	Copysign(0, -1),
	0,
	Inf(1),
	NaN(),
}
var fabsSC = []float32{
	Inf(1),
	0,
	0,
	Inf(1),
	NaN(),
}

var vffdimSC = [][2]float32{
	{Inf(-1), Inf(-1)},
	{Inf(-1), Inf(1)},
	{Inf(-1), NaN()},
	{Copysign(0, -1), Copysign(0, -1)},
	{Copysign(0, -1), 0},
	{0, Copysign(0, -1)},
	{0, 0},
	{Inf(1), Inf(-1)},
	{Inf(1), Inf(1)},
	{Inf(1), NaN()},
	{NaN(), Inf(-1)},
	{NaN(), Copysign(0, -1)},
	{NaN(), 0},
	{NaN(), Inf(1)},
	{NaN(), NaN()},
}
var nan = Float32frombits(0x7FF80001) // SSE2 DIVSD 0/0
var vffdim2SC = [][2]float32{
	{Inf(-1), Inf(-1)},
	{Inf(-1), Inf(1)},
	{Inf(-1), nan},
	{Copysign(0, -1), Copysign(0, -1)},
	{Copysign(0, -1), 0},
	{0, Copysign(0, -1)},
	{0, 0},
	{Inf(1), Inf(-1)},
	{Inf(1), Inf(1)},
	{Inf(1), nan},
	{nan, Inf(-1)},
	{nan, Copysign(0, -1)},
	{nan, 0},
	{nan, Inf(1)},
	{nan, nan},
}
var fdimSC = []float32{
	NaN(),
	0,
	NaN(),
	0,
	0,
	0,
	0,
	Inf(1),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
}
var fmaxSC = []float32{
	Inf(-1),
	Inf(1),
	NaN(),
	Copysign(0, -1),
	0,
	0,
	0,
	Inf(1),
	Inf(1),
	Inf(1),
	NaN(),
	NaN(),
	NaN(),
	Inf(1),
	NaN(),
}
var fminSC = []float32{
	Inf(-1),
	Inf(-1),
	Inf(-1),
	Copysign(0, -1),
	Copysign(0, -1),
	Copysign(0, -1),
	0,
	Inf(-1),
	Inf(1),
	NaN(),
	Inf(-1),
	NaN(),
	NaN(),
	NaN(),
	NaN(),
}

var vffrexpSC = []float32{
	Inf(-1),
	Copysign(0, -1),
	0,
	Inf(1),
	NaN(),
}
var frexpSC = []fi{
	{Inf(-1), 0},
	{Copysign(0, -1), 0},
	{0, 0},
	{Inf(1), 0},
	{NaN(), 0},
}

var vfldexpSC = []fi{
	{0, 0},
	{0, -1075},
	{0, 1024},
	{Copysign(0, -1), 0},
	{Copysign(0, -1), -1075},
	{Copysign(0, -1), 1024},
	{Inf(1), 0},
	{Inf(1), -1024},
	{Inf(-1), 0},
	{Inf(-1), -1024},
	{NaN(), -1024},
}
var ldexpSC = []float32{
	0,
	0,
	0,
	Copysign(0, -1),
	Copysign(0, -1),
	Copysign(0, -1),
	Inf(1),
	Inf(1),
	Inf(-1),
	Inf(-1),
	NaN(),
}

var vflogSC = []float32{
	Inf(-1),
	-Pi,
	Copysign(0, -1),
	0,
	1,
	Inf(1),
	NaN(),
}
var logSC = []float32{
	NaN(),
	NaN(),
	Inf(-1),
	Inf(-1),
	0,
	Inf(1),
	NaN(),
}

var vfmodfSC = []float32{
	Inf(-1),
	Copysign(0, -1),
	Inf(1),
	NaN(),
}
var modfSC = [][2]float32{
	{Inf(-1), NaN()}, // [2]float32{Copysign(0, -1), Inf(-1)},
	{Copysign(0, -1), Copysign(0, -1)},
	{Inf(1), NaN()}, // [2]float32{0, Inf(1)},
	{NaN(), NaN()},
}

var vfpowSC = [][2]float32{
	{Inf(-1), -Pi},
	{Inf(-1), -3},
	{Inf(-1), Copysign(0, -1)},
	{Inf(-1), 0},
	{Inf(-1), 1},
	{Inf(-1), 3},
	{Inf(-1), Pi},
	{Inf(-1), NaN()},

	{-Pi, Inf(-1)},
	{-Pi, -Pi},
	{-Pi, Copysign(0, -1)},
	{-Pi, 0},
	{-Pi, 1},
	{-Pi, Pi},
	{-Pi, Inf(1)},
	{-Pi, NaN()},

	{-1, Inf(-1)},
	{-1, Inf(1)},
	{-1, NaN()},
	{-1 / 2, Inf(-1)},
	{-1 / 2, Inf(1)},
	{Copysign(0, -1), Inf(-1)},
	{Copysign(0, -1), -Pi},
	{Copysign(0, -1), -3},
	{Copysign(0, -1), 3},
	{Copysign(0, -1), Pi},
	{Copysign(0, -1), Inf(1)},

	{0, Inf(-1)},
	{0, -Pi},
	{0, -3},
	{0, Copysign(0, -1)},
	{0, 0},
	{0, 3},
	{0, Pi},
	{0, Inf(1)},
	{0, NaN()},

	{1 / 2, Inf(-1)},
	{1 / 2, Inf(1)},
	{1, Inf(-1)},
	{1, Inf(1)},
	{1, NaN()},

	{Pi, Inf(-1)},
	{Pi, Copysign(0, -1)},
	{Pi, 0},
	{Pi, 1},
	{Pi, Inf(1)},
	{Pi, NaN()},
	{Inf(1), -Pi},
	{Inf(1), Copysign(0, -1)},
	{Inf(1), 0},
	{Inf(1), 1},
	{Inf(1), Pi},
	{Inf(1), NaN()},
	{NaN(), -Pi},
	{NaN(), Copysign(0, -1)},
	{NaN(), 0},
	{NaN(), 1},
	{NaN(), Pi},
	{NaN(), NaN()},
}
var powSC = []float32{
	0,               // pow(-Inf, -Pi)
	Copysign(0, -1), // pow(-Inf, -3)
	1,               // pow(-Inf, -0)
	1,               // pow(-Inf, +0)
	Inf(-1),         // pow(-Inf, 1)
	Inf(-1),         // pow(-Inf, 3)
	Inf(1),          // pow(-Inf, Pi)
	NaN(),           // pow(-Inf, NaN)
	0,               // pow(-Pi, -Inf)
	NaN(),           // pow(-Pi, -Pi)
	1,               // pow(-Pi, -0)
	1,               // pow(-Pi, +0)
	-Pi,             // pow(-Pi, 1)
	NaN(),           // pow(-Pi, Pi)
	Inf(1),          // pow(-Pi, +Inf)
	NaN(),           // pow(-Pi, NaN)
	1,               // pow(-1, -Inf) IEEE 754-2008
	1,               // pow(-1, +Inf) IEEE 754-2008
	NaN(),           // pow(-1, NaN)
	Inf(1),          // pow(-1/2, -Inf)
	0,               // pow(-1/2, +Inf)
	Inf(1),          // pow(-0, -Inf)
	Inf(1),          // pow(-0, -Pi)
	Inf(-1),         // pow(-0, -3) IEEE 754-2008
	Copysign(0, -1), // pow(-0, 3) IEEE 754-2008
	0,               // pow(-0, +Pi)
	0,               // pow(-0, +Inf)
	Inf(1),          // pow(+0, -Inf)
	Inf(1),          // pow(+0, -Pi)
	Inf(1),          // pow(+0, -3)
	1,               // pow(+0, -0)
	1,               // pow(+0, +0)
	0,               // pow(+0, 3)
	0,               // pow(+0, +Pi)
	0,               // pow(+0, +Inf)
	NaN(),           // pow(+0, NaN)
	Inf(1),          // pow(1/2, -Inf)
	0,               // pow(1/2, +Inf)
	1,               // pow(1, -Inf) IEEE 754-2008
	1,               // pow(1, +Inf) IEEE 754-2008
	1,               // pow(1, NaN) IEEE 754-2008
	0,               // pow(+Pi, -Inf)
	1,               // pow(+Pi, -0)
	1,               // pow(+Pi, +0)
	Pi,              // pow(+Pi, 1)
	Inf(1),          // pow(+Pi, +Inf)
	NaN(),           // pow(+Pi, NaN)
	0,               // pow(+Inf, -Pi)
	1,               // pow(+Inf, -0)
	1,               // pow(+Inf, +0)
	Inf(1),          // pow(+Inf, 1)
	Inf(1),          // pow(+Inf, Pi)
	NaN(),           // pow(+Inf, NaN)
	NaN(),           // pow(NaN, -Pi)
	1,               // pow(NaN, -0)
	1,               // pow(NaN, +0)
	NaN(),           // pow(NaN, 1)
	NaN(),           // pow(NaN, +Pi)
	NaN(),           // pow(NaN, NaN)
}

var vfsignbitSC = []float32{
	Inf(-1),
	Copysign(0, -1),
	0,
	Inf(1),
	NaN(),
}
var signbitSC = []bool{
	true,
	true,
	false,
	false,
	false,
}

var vfsqrtSC = []float32{
	Inf(-1),
	-Pi,
	Copysign(0, -1),
	0,
	Inf(1),
	NaN(),
	Float32frombits(2), // subnormal; see https://golang.org/issue/13013
}
var sqrtSC = []float32{
	NaN(),
	NaN(),
	Copysign(0, -1),
	0,
	Inf(1),
	NaN(),
	3.1434555694052576e-162,
}

var vftanhSC = []float32{
	Inf(-1),
	Copysign(0, -1),
	0,
	Inf(1),
	NaN(),
}
var tanhSC = []float32{
	-1,
	Copysign(0, -1),
	0,
	1,
	NaN(),
}

// arguments and expected results for boundary cases
const (
	SmallestNormalFloat32   = 1.1754943508222875079687365e-38 // 1/(2**(127-1))
	LargestSubnormalFloat32 = SmallestNormalFloat32 - SmallestNonzeroFloat32
)

var vffrexpBC = []float32{
	SmallestNormalFloat32,
	LargestSubnormalFloat32,
	SmallestNonzeroFloat32,
	MaxFloat32,
	-SmallestNormalFloat32,
	-LargestSubnormalFloat32,
	-SmallestNonzeroFloat32,
	-MaxFloat32,
}
var frexpBC = []fi{
	{0.5, -125},
	{0.9999999, -126},
	{0.5, -148},
	{0.99999994, 128},
	{-0.5, -125},
	{-0.9999999, -126},
	{-0.5, -148},
	{-0.99999994, 128},
}

var vfldexpBC = []fi{
	{SmallestNormalFloat32, -23},
	{LargestSubnormalFloat32, -22},
	{SmallestNonzeroFloat32, 256},
	{MaxFloat32, -(127 + 149)},
	{1, -150},
	{-1, -150},
	{1, 128},
	{-1, 128},
}
var ldexpBC = []float32{
	SmallestNonzeroFloat32,
	3e-45,        // 2**-148
	1.6225928e32, // 2**130
	3e-45,        // 2**-127
	0,
	Copysign(0, -1),
	Inf(1),
	Inf(-1),
}

func tolerance(a, b, e float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	// note: b is correct (expected) value, a is actual value.
	// make error tolerance a fraction of b, not a.
	if b != 0 {
		e = e * b
		if e < 0 {
			e = -e
		}
	}
	return d < e
}
func isClose(a, b float32) bool   { return tolerance(a, b, 1e-5) } // the number gotten from the cfloat standard. Haskell's Linear package uses 1e-6 for floats
func veryclose(a, b float32) bool { return tolerance(a, b, 1e-6) } // from wiki
func alike(a, b float32) bool {
	switch {
	case IsNaN(a) && IsNaN(b):
		return true
	case a == b:
		return Signbit(a) == Signbit(b)
	case a == 0:
		return tolerance(b, a, 1e-22)
	}
	return false
}

func TestNaN(t *testing.T) {
	f32 := NaN()
	if f32 == f32 {
		t.Fatalf("NaN() returns %g, expected NaN", f32)
	}
	f64 := float64(f32)
	if f64 == f64 {
		t.Fatalf("float64(NaN()) is %g, expected NaN", f64)
	}
}

func TestCeil(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Ceil(vf[i]); ceil[i] != f {
			t.Errorf("Ceil(%g) = %g, want %g", vf[i], f, ceil[i])
		}
	}
	for i := 0; i < len(vfceilSC); i++ {
		if f := Ceil(vfceilSC[i]); !alike(ceilSC[i], f) {
			t.Errorf("Ceil(%g) = %g, want %g", vfceilSC[i], f, ceilSC[i])
		}
	}
}

func TestCopysign(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Copysign(vf[i], -1); copysign[i] != f {
			t.Errorf("Copysign(%g, -1) = %g, want %g", vf[i], f, copysign[i])
		}
	}
	for i := 0; i < len(vf); i++ {
		if f := Copysign(vf[i], 1); -copysign[i] != f {
			t.Errorf("Copysign(%g, 1) = %g, want %g", vf[i], f, -copysign[i])
		}
	}
	for i := 0; i < len(vfcopysignSC); i++ {
		if f := Copysign(vfcopysignSC[i], -1); !alike(copysignSC[i], f) {
			t.Errorf("Copysign(%g, -1) = %g, want %g", vfcopysignSC[i], f, copysignSC[i])
		}
	}
}

func TestExp(t *testing.T) {
	testExp(t, Exp, "Exp")
	// testExp(t, ExpGo, "ExpGo")
}

func testExp(t *testing.T, Exp func(float32) float32, name string) {
	for i := 0; i < len(vf); i++ {
		if f := Exp(vf[i]); !isClose(exp[i], f) {
			t.Errorf("%s(%g) = %g, want %g", name, vf[i], f, exp[i])
		}
	}
	for i := 0; i < len(vfexpSC); i++ {
		if f := Exp(vfexpSC[i]); !alike(expSC[i], f) {
			t.Errorf("%s(%g) = %g, want %g", name, vfexpSC[i], f, expSC[i])
		}
	}
}

func TestAbs(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Abs(vf[i]); fabs[i] != f {
			t.Errorf("Abs(%g) = %g, want %g", vf[i], f, fabs[i])
		}
	}
	for i := 0; i < len(vffabsSC); i++ {
		if f := Abs(vffabsSC[i]); !alike(fabsSC[i], f) {
			t.Errorf("Abs(%g) = %g, want %g", vffabsSC[i], f, fabsSC[i])
		}
	}
}

func TestDim(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Dim(vf[i], 0); fdim[i] != f {
			t.Errorf("Dim(%g, %g) = %g, want %g", vf[i], 0.0, f, fdim[i])
		}
	}
	for i := 0; i < len(vffdimSC); i++ {
		if f := Dim(vffdimSC[i][0], vffdimSC[i][1]); !alike(fdimSC[i], f) {
			t.Errorf("Dim(%g, %g) = %g, want %g", vffdimSC[i][0], vffdimSC[i][1], f, fdimSC[i])
		}
	}
	for i := 0; i < len(vffdim2SC); i++ {
		if f := Dim(vffdim2SC[i][0], vffdim2SC[i][1]); !alike(fdimSC[i], f) {
			t.Errorf("Dim(%g, %g) = %g, want %g", vffdim2SC[i][0], vffdim2SC[i][1], f, fdimSC[i])
		}
	}
}

func TestFloor(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Floor(vf[i]); floor[i] != f {
			t.Errorf("Floor(%g) = %g, want %g", vf[i], f, floor[i])
		}
	}
	for i := 0; i < len(vfceilSC); i++ {
		if f := Floor(vfceilSC[i]); !alike(ceilSC[i], f) {
			t.Errorf("Floor(%g) = %g, want %g", vfceilSC[i], f, ceilSC[i])
		}
	}
}

func TestMax(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Max(vf[i], ceil[i]); ceil[i] != f {
			t.Errorf("Max(%g, %g) = %g, want %g", vf[i], ceil[i], f, ceil[i])
		}
	}
	for i := 0; i < len(vffdimSC); i++ {
		if f := Max(vffdimSC[i][0], vffdimSC[i][1]); !alike(fmaxSC[i], f) {
			t.Errorf("Max(%g, %g) = %g, want %g", vffdimSC[i][0], vffdimSC[i][1], f, fmaxSC[i])
		}
	}
	for i := 0; i < len(vffdim2SC); i++ {
		if f := Max(vffdim2SC[i][0], vffdim2SC[i][1]); !alike(fmaxSC[i], f) {
			t.Errorf("Max(%g, %g) = %g, want %g", vffdim2SC[i][0], vffdim2SC[i][1], f, fmaxSC[i])
		}
	}
}

func TestMin(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Min(vf[i], floor[i]); floor[i] != f {
			t.Errorf("Min(%g, %g) = %g, want %g", vf[i], floor[i], f, floor[i])
		}
	}
	for i := 0; i < len(vffdimSC); i++ {
		if f := Min(vffdimSC[i][0], vffdimSC[i][1]); !alike(fminSC[i], f) {
			t.Errorf("Min(%g, %g) = %g, want %g", vffdimSC[i][0], vffdimSC[i][1], f, fminSC[i])
		}
	}
	for i := 0; i < len(vffdim2SC); i++ {
		if f := Min(vffdim2SC[i][0], vffdim2SC[i][1]); !alike(fminSC[i], f) {
			t.Errorf("Min(%g, %g) = %g, want %g", vffdim2SC[i][0], vffdim2SC[i][1], f, fminSC[i])
		}
	}
}

func TestFrexp(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f, j := Frexp(vf[i]); !veryclose(frexp[i].f, f) || frexp[i].i != j {
			t.Errorf("Frexp(%g) = %g, %d, want %g, %d", vf[i], f, j, frexp[i].f, frexp[i].i)
		}
	}
	for i := 0; i < len(vffrexpSC); i++ {
		if f, j := Frexp(vffrexpSC[i]); !alike(frexpSC[i].f, f) || frexpSC[i].i != j {
			t.Errorf("Frexp(%g) = %g, %d, want %g, %d", vffrexpSC[i], f, j, frexpSC[i].f, frexpSC[i].i)
		}
	}
	for i := 0; i < len(vffrexpBC); i++ {
		if f, j := Frexp(vffrexpBC[i]); !alike(frexpBC[i].f, f) || frexpBC[i].i != j {
			t.Errorf("Frexp(%g) = %g, %d, want %g, %d", vffrexpBC[i], f, j, frexpBC[i].f, frexpBC[i].i)
		}
	}
}

func TestLdexp(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Ldexp(frexp[i].f, frexp[i].i); !veryclose(vf[i], f) {
			t.Errorf("Ldexp(%g, %d) = %g, want %g", frexp[i].f, frexp[i].i, f, vf[i])
		}
	}
	for i := 0; i < len(vffrexpSC); i++ {
		if f := Ldexp(frexpSC[i].f, frexpSC[i].i); !alike(vffrexpSC[i], f) {
			t.Errorf("Ldexp(%g, %d) = %g, want %g", frexpSC[i].f, frexpSC[i].i, f, vffrexpSC[i])
		}
	}
	for i := 0; i < len(vfldexpSC); i++ {
		if f := Ldexp(vfldexpSC[i].f, vfldexpSC[i].i); !alike(ldexpSC[i], f) {
			t.Errorf("Ldexp(%g, %d) = %g, want %g", vfldexpSC[i].f, vfldexpSC[i].i, f, ldexpSC[i])
		}
	}
	for i := 0; i < len(vffrexpBC); i++ {
		if f := Ldexp(frexpBC[i].f, frexpBC[i].i); !alike(vffrexpBC[i], f) {
			t.Errorf("Ldexp(%g, %d) = %g, want %g", frexpBC[i].f, frexpBC[i].i, f, vffrexpBC[i])
		}
	}
	for i := 0; i < len(vfldexpBC); i++ {
		if f := Ldexp(vfldexpBC[i].f, vfldexpBC[i].i); !alike(ldexpBC[i], f) {
			t.Errorf("Ldexp(%g, %d) = %g, want %g", vfldexpBC[i].f, vfldexpBC[i].i, f, ldexpBC[i])
		}
	}
}

func TestLog(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := Abs(vf[i])
		if f := Log(a); !isClose(log[i], f) {
			t.Errorf("Log(%g) = %g, want %g", a, f, log[i])
		}
	}
	if f := Log(10); f != Ln10 {
		t.Errorf("Log(%g) = %g, want %g", 10.0, f, Ln10)
	}
	for i := 0; i < len(vflogSC); i++ {
		if f := Log(vflogSC[i]); !alike(logSC[i], f) {
			t.Errorf("Log(%g) = %g, want %g", vflogSC[i], f, logSC[i])
		}
	}
}

func TestModf(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f, g := Modf(vf[i]); !veryclose(modf[i][0], f) || !tolerance(modf[i][1], g, 1e-4) {
			t.Errorf("Modf(%g) = %g, %g, want %g, %g", vf[i], f, g, modf[i][0], modf[i][1])
		}
	}
	for i := 0; i < len(vfmodfSC); i++ {
		if f, g := Modf(vfmodfSC[i]); !alike(modfSC[i][0], f) || !alike(modfSC[i][1], g) {
			t.Errorf("Modf(%g) = %g, %g, want %g, %g", vfmodfSC[i], f, g, modfSC[i][0], modfSC[i][1])
		}
	}
}

func TestPow(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Pow(10, vf[i]); !isClose(pow[i], f) {
			t.Errorf("Pow(10, %g) = %g, want %g", vf[i], f, pow[i])
		}
	}
	for i := 0; i < len(vfpowSC); i++ {
		if f := Pow(vfpowSC[i][0], vfpowSC[i][1]); !alike(powSC[i], f) {
			t.Errorf("Pow(%g, %g) = %g, want %g", vfpowSC[i][0], vfpowSC[i][1], f, powSC[i])
		}
	}
}

func TestSignbit(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Signbit(vf[i]); signbit[i] != f {
			t.Errorf("Signbit(%g) = %t, want %t", vf[i], f, signbit[i])
		}
	}
	for i := 0; i < len(vfsignbitSC); i++ {
		if f := Signbit(vfsignbitSC[i]); signbitSC[i] != f {
			t.Errorf("Signbit(%g) = %t, want %t", vfsignbitSC[i], f, signbitSC[i])
		}
	}
}

func TestSqrt(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := Abs(vf[i])
		if f := SqrtGo(a); !veryclose(sqrt[i], f) {
			t.Errorf("SqrtGo(%g) = %g, want %g", a, f, sqrt[i])
		}
		a = Abs(vf[i])
		if f := Sqrt(a); !veryclose(sqrt[i], f) {
			t.Errorf("Sqrt(%g) = %g, want %g", a, f, sqrt[i])
		}
	}
	for i := 0; i < len(vfsqrtSC); i++ {
		if f := SqrtGo(vfsqrtSC[i]); !alike(sqrtSC[i], f) {
			t.Errorf("SqrtGo(%g) = %g, want %g", vfsqrtSC[i], f, sqrtSC[i])
		}
		if f := Sqrt(vfsqrtSC[i]); !alike(sqrtSC[i], f) {
			t.Errorf("Sqrt(%g) = %g, want %g", vfsqrtSC[i], f, sqrtSC[i])
		}
	}
}

func TestTanh(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Tanh(vf[i]); !veryclose(tanh[i], f) {
			t.Errorf("Tanh(%g) = %g, want %g", vf[i], f, tanh[i])
		}
	}
	for i := 0; i < len(vftanhSC); i++ {
		if f := Tanh(vftanhSC[i]); !alike(tanhSC[i], f) {
			t.Errorf("Tanh(%g) = %g, want %g", vftanhSC[i], f, tanhSC[i])
		}
	}
}

func TestTrunc(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := Trunc(vf[i]); trunc[i] != f {
			t.Errorf("Trunc(%g) = %g, want %g", vf[i], f, trunc[i])
		}
	}
	for i := 0; i < len(vfceilSC); i++ {
		if f := Trunc(vfceilSC[i]); !alike(ceilSC[i], f) {
			t.Errorf("Trunc(%g) = %g, want %g", vfceilSC[i], f, ceilSC[i])
		}
	}
}

// Check that math constants are accepted by compiler
// and have right value (assumes strconv.ParseFloat works).
// https://golang.org/issue/201

type floatTest struct {
	val  interface{}
	name string
	str  string
}

var floatTests = []floatTest{
	{float32(MaxFloat32), "MaxFloat32", "3.4028235e+38"},
	{float32(SmallestNonzeroFloat32), "SmallestNonzeroFloat32", "1e-45"},
}

func TestFloatMinMax(t *testing.T) {
	for _, tt := range floatTests {
		s := fmt.Sprint(tt.val)
		if s != tt.str {
			t.Errorf("Sprint(%v) = %s, want %s", tt.name, s, tt.str)
		}
	}
}

// Benchmarks

func BenchmarkCeil(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Ceil(.5)
	}
}

func BenchmarkCopysign(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Copysign(.5, -1)
	}
}

func BenchmarkExp(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Exp(.5)
	}
}

func BenchmarkAbs(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Abs(.5)
	}
}

func BenchmarkDim(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Dim(10, 3)
	}
}

func BenchmarkFloor(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Floor(.5)
	}
}

func BenchmarkMax(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Max(10, 3)
	}
}

func BenchmarkMin(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Min(10, 3)
	}
}

func BenchmarkFrexp(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Frexp(8)
	}
}

func BenchmarkLdexp(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Ldexp(.5, 2)
	}
}

func BenchmarkLog(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Log(.5)
	}
}

func BenchmarkModf(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Modf(1.5)
	}
}

func BenchmarkPowInt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Pow(2, 2)
	}
}

func BenchmarkPowFrac(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Pow(2.5, 1.5)
	}
}

func BenchmarkSignbit(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Signbit(2.5)
	}
}

func BenchmarkSqrt(b *testing.B) {
	var x, y float32
	x, y = 0.0, 10.0
	for i := 0; i < b.N; i++ {
		x += Sqrt(y)
	}
}

func BenchmarkSqrtIndirect(b *testing.B) {
	var x, y float32
	x, y = 0.0, 10.0
	f := Sqrt
	for i := 0; i < b.N; i++ {
		x += f(y)
	}
}

func isPrime(i int) bool {
	// Yes, this is a dumb way to write this code,
	// but calling Sqrt repeatedly in this way demonstrates
	// the benefit of using a direct SQRT instruction on systems
	// that have one, whereas the obvious loop seems not to
	// demonstrate such a benefit.
	for j := 2; float32(j) <= Sqrt(float32(i)); j++ {
		if i%j == 0 {
			return false
		}
	}
	return true
}

func BenchmarkSqrtPrime(b *testing.B) {
	for i := 0; i < b.N; i++ {
		isPrime(100003)
	}
}

func BenchmarkTanh(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Tanh(2.5)
	}
}
func BenchmarkTrunc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Trunc(.5)
	}
}
