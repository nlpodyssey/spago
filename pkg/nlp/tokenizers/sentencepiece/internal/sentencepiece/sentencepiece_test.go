// Copyright (c) 2020 Vikesh Raj C. All rights reserved.
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

package sentencepiece

import (
	"reflect"
	"testing"
)

func TestTokenization(t *testing.T) {
	sp, err := NewSentencepieceFromFile("test_data/xlnet-base-cased-spiece.model", false)
	if err != nil {
		t.Errorf("Unable to create sentencepiece")
		return
	}

	tests := []struct {
		text   string
		tokens []Token
	}{
		{text: "this", tokens: []Token{{ID: 52, Text: "▁this"}}},
		{text: "hello", tokens: []Token{{ID: 24717, Text: "▁hello"}}},
		{text: "This is a sample sentence to be tokénized", tokens: []Token{
			{ID: 122, Text: "▁This"},
			{ID: 27, Text: "▁is"},
			{ID: 24, Text: "▁a"},
			{ID: 4561, Text: "▁sample"},
			{ID: 3833, Text: "▁sentence"},
			{ID: 22, Text: "▁to"},
			{ID: 39, Text: "▁be"},
			{ID: 22, Text: "▁to"},
			{ID: 267, Text: "k"},
			{ID: 0, Text: "é"},
			{ID: 180, Text: "n"},
			{ID: 1227, Text: "ized"},
		}},
		{text: "Wondering how this will get tokenized 🤔 ?", tokens: []Token{
			{ID: 14748, Text: "▁Wonder"},
			{ID: 56, Text: "ing"},
			{ID: 160, Text: "▁how"},
			{ID: 52, Text: "▁this"},
			{ID: 53, Text: "▁will"},
			{ID: 133, Text: "▁get"},
			{ID: 17366, Text: "▁token"},
			{ID: 1227, Text: "ized"},
			{ID: 17, Text: "▁"},
			{ID: 0, Text: "🤔"},
			{ID: 17, Text: "▁"},
			{ID: 82, Text: "?"},
		}},
		{text: "İs th!s 𩸽 Ϻ Šœ Ugljšić dấu nặng", tokens: []Token{
			{ID: 17, Text: "▁"},
			{ID: 0, Text: "İ"},
			{ID: 23, Text: "s"},
			{ID: 17, Text: "▁"},
			{ID: 138, Text: "th"},
			{ID: 136, Text: "!"},
			{ID: 23, Text: "s"},
			{ID: 17, Text: "▁"},
			{ID: 0, Text: "𩸽"},
			{ID: 17, Text: "▁"},
			{ID: 0, Text: "Ϻ"},
			{ID: 17, Text: "▁"},
			{ID: 0, Text: "Š"},
			{ID: 128, Text: "▁U"},
			{ID: 15222, Text: "gl"},
			{ID: 1315, Text: "j"},
			{ID: 0, Text: "š"},
			{ID: 150, Text: "i"},
			{ID: 0, Text: "ć"},
			{ID: 17, Text: "▁"},
			{ID: 66, Text: "d"},
			{ID: 0, Text: "ấ"},
			{ID: 660, Text: "u"},
			{ID: 17, Text: "▁"},
			{ID: 180, Text: "n"},
			{ID: 0, Text: "ặ"},
			{ID: 3511, Text: "ng"},
		}},
		{text: "compose email to john saying i will be running late to office today because i am not feeling well, my head is aching and in the body add shall we meet next week and when we go to the office lets reach by around 10 am and go for a movie in the evening, may be Spiderman which seems to be a very good movie which got 5 star review from rottentomatoes and imdb", tokens: []Token{
			{ID: 23391, Text: "▁compose"},
			{ID: 1706, Text: "▁email"},
			{ID: 22, Text: "▁to"},
			{ID: 17, Text: "▁"},
			{ID: 22116, Text: "john"},
			{ID: 591, Text: "▁saying"},
			{ID: 17, Text: "▁"},
			{ID: 150, Text: "i"},
			{ID: 53, Text: "▁will"},
			{ID: 39, Text: "▁be"},
			{ID: 926, Text: "▁running"},
			{ID: 471, Text: "▁late"},
			{ID: 22, Text: "▁to"},
			{ID: 495, Text: "▁office"},
			{ID: 494, Text: "▁today"},
			{ID: 149, Text: "▁because"},
			{ID: 17, Text: "▁"},
			{ID: 150, Text: "i"},
			{ID: 569, Text: "▁am"},
			{ID: 50, Text: "▁not"},
			{ID: 1803, Text: "▁feeling"},
			{ID: 143, Text: "▁well"},
			{ID: 19, Text: ","},
			{ID: 94, Text: "▁my"},
			{ID: 291, Text: "▁head"},
			{ID: 27, Text: "▁is"},
			{ID: 24, Text: "▁a"},
			{ID: 5410, Text: "ching"},
			{ID: 21, Text: "▁and"},
			{ID: 25, Text: "▁in"},
			{ID: 18, Text: "▁the"},
			{ID: 458, Text: "▁body"},
			{ID: 1319, Text: "▁add"},
			{ID: 1530, Text: "▁shall"},
			{ID: 80, Text: "▁we"},
			{ID: 767, Text: "▁meet"},
			{ID: 244, Text: "▁next"},
			{ID: 260, Text: "▁week"},
			{ID: 21, Text: "▁and"},
			{ID: 90, Text: "▁when"},
			{ID: 80, Text: "▁we"},
			{ID: 216, Text: "▁go"},
			{ID: 22, Text: "▁to"},
			{ID: 18, Text: "▁the"},
			{ID: 495, Text: "▁office"},
			{ID: 10234, Text: "▁lets"},
			{ID: 1287, Text: "▁reach"},
			{ID: 37, Text: "▁by"},
			{ID: 199, Text: "▁around"},
			{ID: 241, Text: "▁10"},
			{ID: 569, Text: "▁am"},
			{ID: 21, Text: "▁and"},
			{ID: 216, Text: "▁go"},
			{ID: 28, Text: "▁for"},
			{ID: 24, Text: "▁a"},
			{ID: 1432, Text: "▁movie"},
			{ID: 25, Text: "▁in"},
			{ID: 18, Text: "▁the"},
			{ID: 2060, Text: "▁evening"},
			{ID: 19, Text: ","},
			{ID: 132, Text: "▁may"},
			{ID: 39, Text: "▁be"},
			{ID: 17489, Text: "▁Spider"},
			{ID: 249, Text: "man"},
			{ID: 59, Text: "▁which"},
			{ID: 1303, Text: "▁seems"},
			{ID: 22, Text: "▁to"},
			{ID: 39, Text: "▁be"},
			{ID: 24, Text: "▁a"},
			{ID: 172, Text: "▁very"},
			{ID: 195, Text: "▁good"},
			{ID: 1432, Text: "▁movie"},
			{ID: 59, Text: "▁which"},
			{ID: 345, Text: "▁got"},
			{ID: 306, Text: "▁5"},
			{ID: 1795, Text: "▁star"},
			{ID: 1398, Text: "▁review"},
			{ID: 40, Text: "▁from"},
			{ID: 28626, Text: "▁rotten"},
			{ID: 261, Text: "to"},
			{ID: 18693, Text: "mato"},
			{ID: 202, Text: "es"},
			{ID: 21, Text: "▁and"},
			{ID: 7693, Text: "▁im"},
			{ID: 66, Text: "d"},
			{ID: 508, Text: "b"},
		}},
	}

	for _, test := range tests {
		output := sp.Tokenize(test.text)
		if !reflect.DeepEqual(output, test.tokens) {
			t.Errorf("Tokenization error : %s, len %d, got %v || expected %v", test.text, len(test.text), output, test.tokens)
		}
	}
}

func TestTokenizationSPM(t *testing.T) {
	sp, err := NewSentencepieceFromFile("test_data/spm.model", true)
	if err != nil {
		t.Errorf("Unable to create sentencepiece")
		return
	}

	tests := []struct {
		text   string
		tokens []Token
	}{
		{text: "this", tokens: []Token{{ID: 48, Text: "▁this"}}},
		{text: "hello", tokens: []Token{{ID: 10975, Text: "▁hello"}}},
		{text: "This is a sample sentence to be tokénized", tokens: []Token{
			{ID: 48, Text: "▁this"},
			{ID: 25, Text: "▁is"},
			{ID: 21, Text: "▁a"},
			{ID: 5717, Text: "▁sample"},
			{ID: 5123, Text: "▁sentence"},
			{ID: 20, Text: "▁to"},
			{ID: 44, Text: "▁be"},
			{ID: 20, Text: "▁to"},
			{ID: 197, Text: "k"},
			{ID: 1, Text: "é"},
			{ID: 103, Text: "n"},
			{ID: 1333, Text: "ized"},
		}},
		{text: ".", tokens: []Token{{ID: 13, Text: "▁"}, {ID: 9, Text: "."}}},
		{text: "this is a dot .", tokens: []Token{
			{ID: 48, Text: "▁this"},
			{ID: 25, Text: "▁is"},
			{ID: 21, Text: "▁a"},
			{ID: 14123, Text: "▁dot"},
			{ID: 13, Text: "▁"},
			{ID: 9, Text: "."},
		}},
		{text: "compose email to john saying i will be running late to office today because i am not feeling well, my head is aching and in the body add shall we meet next week and when we go to the office lets reach by around 10 am and go for a movie in the evening, may be Spiderman which seems to be a very good movie which got 5 star review from rottentomatoes and imdb", tokens: []Token{
			{ID: 18217, Text: "▁compose"},
			{ID: 8517, Text: "▁email"},
			{ID: 20, Text: "▁to"},
			{ID: 239, Text: "▁john"},
			{ID: 1148, Text: "▁saying"},
			{ID: 31, Text: "▁i"},
			{ID: 129, Text: "▁will"},
			{ID: 44, Text: "▁be"},
			{ID: 946, Text: "▁running"},
			{ID: 456, Text: "▁late"},
			{ID: 20, Text: "▁to"},
			{ID: 488, Text: "▁office"},
			{ID: 786, Text: "▁today"},
			{ID: 185, Text: "▁because"},
			{ID: 31, Text: "▁i"},
			{ID: 589, Text: "▁am"},
			{ID: 52, Text: "▁not"},
			{ID: 1249, Text: "▁feeling"},
			{ID: 134, Text: "▁well"},
			{ID: 15, Text: ","},
			{ID: 51, Text: "▁my"},
			{ID: 157, Text: "▁head"},
			{ID: 25, Text: "▁is"},
			{ID: 17010, Text: "▁aching"},
			{ID: 17, Text: "▁and"},
			{ID: 19, Text: "▁in"},
			{ID: 14, Text: "▁the"},
			{ID: 358, Text: "▁body"},
			{ID: 3547, Text: "▁add"},
			{ID: 3004, Text: "▁shall"},
			{ID: 95, Text: "▁we"},
			{ID: 1255, Text: "▁meet"},
			{ID: 328, Text: "▁next"},
			{ID: 877, Text: "▁week"},
			{ID: 17, Text: "▁and"},
			{ID: 76, Text: "▁when"},
			{ID: 95, Text: "▁we"},
			{ID: 162, Text: "▁go"},
			{ID: 20, Text: "▁to"},
			{ID: 14, Text: "▁the"},
			{ID: 488, Text: "▁office"},
			{ID: 6884, Text: "▁lets"},
			{ID: 1470, Text: "▁reach"},
			{ID: 34, Text: "▁by"},
			{ID: 140, Text: "▁around"},
			{ID: 332, Text: "▁10"},
			{ID: 589, Text: "▁am"},
			{ID: 17, Text: "▁and"},
			{ID: 162, Text: "▁go"},
			{ID: 26, Text: "▁for"},
			{ID: 21, Text: "▁a"},
			{ID: 1308, Text: "▁movie"},
			{ID: 19, Text: "▁in"},
			{ID: 14, Text: "▁the"},
			{ID: 2089, Text: "▁evening"},
			{ID: 15, Text: ","},
			{ID: 123, Text: "▁may"},
			{ID: 44, Text: "▁be"},
			{ID: 5650, Text: "▁spider"},
			{ID: 177, Text: "man"},
			{ID: 56, Text: "▁which"},
			{ID: 2206, Text: "▁seems"},
			{ID: 20, Text: "▁to"},
			{ID: 44, Text: "▁be"},
			{ID: 21, Text: "▁a"},
			{ID: 253, Text: "▁very"},
			{ID: 254, Text: "▁good"},
			{ID: 1308, Text: "▁movie"},
			{ID: 56, Text: "▁which"},
			{ID: 330, Text: "▁got"},
			{ID: 331, Text: "▁5"},
			{ID: 778, Text: "▁star"},
			{ID: 1487, Text: "▁review"},
			{ID: 37, Text: "▁from"},
			{ID: 11573, Text: "▁rotten"},
			{ID: 262, Text: "to"},
			{ID: 8844, Text: "mato"},
			{ID: 160, Text: "es"},
			{ID: 17, Text: "▁and"},
			{ID: 797, Text: "▁im"},
			{ID: 9007, Text: "db"},
		}},
	}

	for _, test := range tests {
		output := sp.Tokenize(test.text)
		if !reflect.DeepEqual(output, test.tokens) {
			t.Errorf("Tokenization error : %s, len %d, got %v || expected %v", test.text, len(test.text), output, test.tokens)
		}
	}
}

func TestControlWords(t *testing.T) {
	sp, err := NewSentencepieceFromFile("test_data/xlnet-base-cased-spiece.model", false)
	if err != nil {
		t.Errorf("Unable to create sentencepiece")
		return
	}

	unknownIndex := sp.GetUnknownIndex()
	if unknownIndex != 0 {
		t.Errorf("Unknown index not equal to 0")
	}

	clsIndex, ok := sp.GetControlWord("<cls>")
	if !ok || clsIndex != 3 {
		t.Errorf("Control word [CLS] not correct : %d", clsIndex)
	}

}

func TestControlWords2(t *testing.T) {
	sp, err := NewSentencepieceFromFile("test_data/spm.model", true)
	if err != nil {
		t.Errorf("Unable to create sentencepiece")
		return
	}

	unknownIndex := sp.GetUnknownIndex()
	if unknownIndex != 1 {
		t.Errorf("Unknown index not equal to 1")
	}

	clsIndex, ok := sp.GetControlWord("[CLS]")
	if !ok || clsIndex != 2 {
		t.Errorf("Control word [CLS] not correct")
	}
}

func BenchmarkSentencePiece(b *testing.B) {
	sp, err := NewSentencepieceFromFile("test_data/xlnet-base-cased-spiece.model", false)
	if err != nil {
		b.Errorf("Unable to create sentencepiece")
		return
	}

	b.ResetTimer()

	inputs := []string{
		"compose email to john saying i will be running late to office today because i am not feeling well, my head is aching and in the body add shall we meet next week and when we go to the office lets reach by around 10 am and go for a movie in the evening, may be Spiderman which seems to be a very good movie which got 5 star review from rottentomatoes and imdb",
	}

	for _, input := range inputs {
		b.Run(firstNChars(input, 20), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sp.Tokenize(input)
			}
		})
	}
}

func firstNChars(s string, n int) string {
	if len(s) < n {
		return s
	}
	return s[:n]
}
