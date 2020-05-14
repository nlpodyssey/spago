// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"bufio"
	"bytes"
	"github.com/gosuri/uiprogress"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/pkg/utils/kvdb"
	"log"
	"os"
	"sync"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

var allModels []*Model

// TODO: add dedicated embeddings for out-of-vocabulary (OOV) words and other special words
type Model struct {
	Config
	storage        kvdb.KeyValueDB
	mu             sync.Mutex
	UsedEmbeddings map[string]*nn.Param `type:"weights"`
}

// TODO: add Dropout
type Config struct {
	// Size of the embedding vectors
	Size int
	// The path to DB on the drive
	DBPath string
	// Whether to use the map in read-only mode (embeddings are not updated during training).
	ReadOnly bool
	// Whether to force the deletion of any existing DB to start with an empty embeddings map
	ForceNewDB bool
}

// New returns a new embedding model.
func New(config Config) *Model {
	m := &Model{
		Config: config,
		storage: kvdb.NewDefaultKeyValueDB(kvdb.Config{
			Path:     config.DBPath,
			ReadOnly: config.ReadOnly,
			ForceNew: config.ForceNewDB,
		}),
		UsedEmbeddings: map[string]*nn.Param{},
	}
	allModels = append(allModels, m)
	return m
}

// Close closes the DB underlying the model of the embeddings map.
// It automatically clears the cache.
func (m *Model) Close() {
	_ = m.storage.Close() // explicitly ignore errors here
	m.ClearUsedEmbeddings()
}

// ClearUsedEmbeddings clears the cache of the used embeddings.
// Beware of any external references to the values of m.UsedEmbeddings. These are weak references!
func (m *Model) ClearUsedEmbeddings() {
	m.mu.Lock()
	m.UsedEmbeddings = map[string]*nn.Param{}
	m.mu.Unlock()
}

// Close closes the DBs underlying all instantiated embeddings models.
// It automatically clears the caches.
func Close() {
	for _, model := range allModels {
		model.Close()
	}
}

// ClearUsedEmbeddings clears the cache of the used embeddings of all instantiated embeddings models.
// Beware of any external references to the values of m.UsedEmbeddings. These are weak references!
func ClearUsedEmbeddings() {
	for _, model := range allModels {
		model.ClearUsedEmbeddings()
	}
}

// SetEmbeddings inserts a new word embeddings.
// If the word is already on the map, overwrites the existing value with the new one.
func (m *Model) SetEmbedding(word string, value *mat.Dense) {
	if m.ReadOnly {
		log.Fatal("embedding: set operation not permitted in read-only mode")
	}
	embedding := nn.NewParam(value)
	embedding.SetPayload(nn.NewEmptySupport())
	var buf bytes.Buffer
	if _, err := (&nn.ParamSerializer{Param: embedding}).Serialize(&buf); err != nil {
		log.Fatal(err)
	}
	if err := m.storage.Put([]byte(word), buf.Bytes()); err != nil {
		log.Fatal(err)
	}
}

// GetEmbedding returns the parameter (the word embedding) associated with the given word.
// The returned embedding is also cached in m.UsedEmbeddings for two reasons:
//     - to allow a faster recovery;
//     - to keep track of used embeddings, should they be optimized.
// If no embedding is found, nil is returned.
// It panics in case of storage errors.
func (m *Model) GetEmbedding(word string) *nn.Param {
	if embedding, ok := m.UsedEmbeddings[word]; ok {
		return embedding
	}
	data, ok, err := m.storage.Get([]byte(word))
	if err != nil {
		log.Fatal(err)
	}
	if !ok {
		return nil // embedding not found
	}
	embedding := nn.NewParam(nil, nn.SetStorage(m.storage))
	if _, err := (&nn.ParamSerializer{Param: embedding}).Deserialize(bytes.NewReader(data)); err != nil {
		log.Fatal(err)
	}
	if m.ReadOnly {
		nn.RequiresGrad(false)(embedding)
	}
	embedding.SetName(word)
	m.mu.Lock()
	m.UsedEmbeddings[word] = embedding // important
	m.mu.Unlock()
	return embedding
}

// Load inserts the pre-trained embeddings into the model.
func (m *Model) Load(filename string) {
	count, err := utils.CountLines(filename)
	if err != nil {
		log.Fatal(err)
	}

	uip := uiprogress.New()
	bar := uip.AddBar(count)
	bar.AppendCompleted().PrependElapsed()
	uip.Start() // start bar rendering
	defer uip.Stop()

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lineCount := 0
	for scanner.Scan() {
		lineCount++
		bar.Incr()
		if lineCount > 1 { // skip header
			line := scanner.Text()
			key := utils.BeforeSpace(line)
			strVec := utils.AfterSpace(line)
			data, err := f64utils.StrToFloat64Slice(strVec)
			if err != nil {
				log.Fatal(err)
			}
			m.SetEmbedding(key, mat.NewVecDense(data))
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}

type Processor struct {
	opt            []interface{}
	model          *Model
	mode           nn.ProcessingMode
	g              *ag.Graph
	UsedEmbeddings map[string]*nn.Param
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model: m,
		mode:  nn.Training,
		opt:   opt,
		g:     g,
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("embeddings: invalid init options")
	}
}

func (p *Processor) Model() nn.Model                { return p.model }
func (p *Processor) Graph() *ag.Graph               { return p.g }
func (p *Processor) RequiresFullSeq() bool          { return false }
func (p *Processor) Mode() nn.ProcessingMode        { return p.mode }
func (p *Processor) SetMode(mode nn.ProcessingMode) { p.mode = mode }

// Encodes returns the embeddings associated with the input words.
// The embeddings are returned as Node(s) already inserted in the graph.
// To words that have no embeddings, the corresponding nodes are nil.
func (p *Processor) Encode(words ...string) []ag.Node {
	encoding := make([]ag.Node, len(words))
	cache := make(map[string]ag.Node) // be smart, don't create two nodes for the same word!
	for i, word := range words {
		if item, ok := cache[word]; ok {
			encoding[i] = item
		} else {
			embedding := p.model.GetEmbedding(word)
			if embedding != nil {
				encoding[i] = p.g.NewWrap(embedding)
			} else {
				encoding[i] = nil
			}
			cache[word] = encoding[i]
		}
	}
	return encoding
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("embeddings: p.Forward() not implemented. Use p.Encode() instead.")
}
