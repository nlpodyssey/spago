// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"fmt"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/gdmbuilder"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/pkg/utils"
	"io"
	"log"
	"os"
	"runtime"
)

// TrainingConfig provides configuration settings for a BERT Trainer.
type TrainingConfig struct {
	Seed             uint64
	BatchSize        int
	GradientClipping mat.Float
	UpdateMethod     gd.MethodConfig
	CorpusPath       string
	ModelPath        string
}

// Trainer implements the training process for a BERT Model.
type Trainer struct {
	TrainingConfig
	randGen       *rand.LockedRand
	optimizer     *gd.GradientDescent
	bestLoss      mat.Float
	lastBatchLoss mat.Float
	model         *Model
	countLine     int
}

// NewTrainer returns a new BERT Trainer.
func NewTrainer(model *Model, config TrainingConfig) *Trainer {
	optimizer := gd.NewOptimizer(gdmbuilder.NewMethod(config.UpdateMethod), nn.NewDefaultParamsIterator(model))
	if config.GradientClipping != 0.0 {
		gd.ClipGradByNorm(config.GradientClipping, 2.0)(optimizer)
	}
	return &Trainer{
		TrainingConfig: config,
		randGen:        rand.NewLockedRand(config.Seed),
		optimizer:      optimizer,
		model:          model,
	}
}

// Train executes the training process.
func (t *Trainer) Train() {
	t.forEachLine(func(i int, text string) {
		t.trainPassage(text)
		t.optimizer.IncBatch()
		t.optimizer.IncExample()
		t.optimizer.Optimize()

		if i > 0 && i%1000 == 0 {
			fmt.Println("=== MODEL SERIALIZATION")
			err := utils.SerializeToFile(t.ModelPath, t.model)
			if err != nil {
				panic("bert: error during model serialization.")
			}
		}

		t.countLine++
	})
}

func (t *Trainer) tokenize(text string) []string {
	tokenizer := wordpiecetokenizer.New(t.model.Vocabulary)
	tokenized := append(tokenizers.GetStrings(tokenizer.Tokenize(text)), wordpiecetokenizer.DefaultSequenceSeparator)
	return append([]string{wordpiecetokenizer.DefaultClassToken}, tokenized...)
}

func (t *Trainer) trainPassage(text string) {
	tokenized := t.tokenize(text)
	if len(tokenized) > t.model.Embeddings.MaxPositions {
		return // skip, sequence too long
	}

	g := ag.NewGraph(ag.Rand(t.randGen), ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	proc := nn.ReifyForTraining(t.model, g).(*Model)

	maskedTokens, maskedIds := t.applyMask(tokenized)
	if len(maskedIds) == 0 {
		return // skip, nothing to learn
	}

	encoded := proc.Encode(maskedTokens)
	predicted := proc.PredictMasked(encoded, maskedIds)

	var loss ag.Node
	for _, id := range maskedIds {
		target, _ := t.model.Vocabulary.ID(tokenized[id])
		loss = g.Add(loss, losses.CrossEntropy(g, predicted[id], target))
	}
	if loss == nil {
		panic("bert: expected loss not to be nil")
	}

	g.Backward(loss)
	t.lastBatchLoss = loss.ScalarValue()
	fmt.Printf("Cnt: %d Loss: %.6f\n", t.countLine, t.lastBatchLoss)
}

func (t *Trainer) applyMask(tokens []string) (newTokens []string, maskedIds []int) {
	for id, word := range tokens {
		if wordpiecetokenizer.IsDefaultSpecial(word) { // don't mask special tokens
			newTokens = append(newTokens, word)
			continue
		}
		if t.randGen.Float() < 0.15 {
			maskedIds = append(maskedIds, id)
			newTokens = append(newTokens, t.getMaskedForm(word))
		} else {
			newTokens = append(newTokens, word)
		}
	}
	return
}

func (t *Trainer) getMaskedForm(orig string) string {
	prob := t.randGen.Float()
	switch {
	case prob < 0.80:
		return wordpiecetokenizer.DefaultMaskToken
	case prob < 0.90:
		randomID := int(mat.Floor(t.randGen.Float() * mat.Float(t.model.Vocabulary.Size())))
		newWord, _ := t.model.Vocabulary.Term(randomID)
		return newWord
	default:
		return orig
	}
}

func (t *Trainer) forEachLine(callback func(i int, line string)) {
	f, err := os.Open(t.CorpusPath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	uncompressedStream, err := gzip.NewReader(f)
	if err != nil {
		log.Fatal(err)
	}
	tarReader := tar.NewReader(uncompressedStream)
	i := 0
	for true {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("Next() failed: %s", err.Error())
		}
		if header.Typeflag == tar.TypeReg {
			scanner := bufio.NewScanner(tarReader)
			for scanner.Scan() {
				i++
				callback(i, scanner.Text())
			}
			if err := scanner.Err(); err != nil {
				log.Fatal(err)
			}
		}
	}
}
