# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Require Go version `1.17`.
- Updating initialization of BatchNorm normalization, making training more
  stable at the beginning. Initializing the weight matrix with a small nonzero
  value improves the behaviour of gradients and stabilizes training.
- Dependencies upgrade.

### Removed
- Global heap allocation "ballast" and math optimization level.

## [0.7.0] - 2021-05-24glg

### Added
- New package `ml/ag/encoding/dot`, for simple serialization of a Graph to 
  DOT (Graphviz) format.
- New package `ml/nn/sgu`, implementing a Spatial Gating Unit (SGU) model.
- New package `ml/nn/conv1x1`, implementing a simple 1-dimensional
  1-sized-kernel convolution model.
- New package `ml/nn/gmlp`, implementing a [gMLP](https://arxiv.org/pdf/2105.08050.pdf)
  model.

### Changed
- `ml/nn/activation/Model.Forward` now simply returns the input as it is if
  the activation function is the identity.

## [0.6.0] - 2021-05-13

### Added
- `ml/losses.WeightedCrossEntropy()`
- `ml/losses.FocalLoss()`
- `ml/losses.WeightedFocalLoss()`
- `nlp/sequencelabeler.LoadModel()` (it replaces `Load()` and `LoadEmbeddings()`)
- `nlp/charlm.LoadModel()`
- `nlp/transformers/bert.Model.PredictMLM()`
- `nlp/transformers/bart/tasks` package
- `nlp/transformers/bert.Model.Vectorize()`
- `ml/ag.Graph.Nodes()` and `ml/ag.Nodes()`
- `ml/nn.Model.Close()`
- `ml/nn.ReifyForTraining()` and `ml/nn.ReifyForInference()`
- `ml/ag.Graph.Backward()` now panics if it is executed with nodes belonging to
  different graphs.
- The new `ml/graphviz` package allows exporting a Graph to [Graphviz](https://graphviz.org/)
  [DOT](https://graphviz.org/pdf/dotguide.pdf) format. To make it possible,
  we introduced a new go-mod dependency [gographviz](https://github.com/awalterschulze/gographviz).
- A custom name can be optionally set to a Graph's Variables. This can be
  useful for debugging purposes and visual graph representation.
  You can now use `Graph.NewVariableWithName()` and `Graph.NewScalarWithName()`
  to create named Variables, and get the name of a Variable with
  `Variable.Name()`.

### Changed
- All `UnaryElementwise` functions provided by the package `ag/fn` have been
  promoted to separate dedicated structs. This improves debuggability and you
  can get appropriate function names when using reflection. Here is the full
  list of the modified functions: `Tan`, `Tanh`, `Sigmoid`, `HardSigmoid`,
  `HardTanh`, `ReLU`, `Softsign`, `Cos`, `Sin`, `Exp`, `Log`, `Neg`,
  `Reciprocal`, `Abs`, `Mish`, `GELU`, `Sqrt`, `Swish`.
  For the same reason, a dedicated `Square` function is introduced, replacing
  `Prod` with both operands set to the same value.
- `ml/ag` types `Operator`, `Variable`, `Wrapper` are now public.
- `ml/nn.Reify()` now expects a Graph and a Processing Mode arguments
  instead of a `Context` object (removed).
- `ml/nn.BaseModel` has been modified, replacing the field `Ctx Context` with
  a direct reference to the model's Graph and the Processing Mode (fields `G`
  and `ProcessingMode`).
- Refactoring server implementation of `nlp/sequencelabeler`,
  `nlp/transformers/bert`, and `nlp/transformers/bart`.
- Upgrade various dependencies.
- Regenerate protocol buffers files (with `protoc-gen-go` v1.26.0 and
  `protoc` v3.16.0).

### Removed
- `nlp/sequencelabeler.Load()` and `LoadEmbeddings()` (now replaced by
  `nlp/sequencelabeler.LoadModel()`)
- `ml/nn.Context` (see related changes on `Reify()` and `BaseModel`)

## [0.5.2] - 2021-03-16

### Added

- Handle multiple BERT pooling strategies (e.g. CLS_TOKEN, REDUCE_MEAN) in `nlp.transformers.bert.server_encode.go`.

## [0.5.1] - 2021-03-07

### Added

- Add `nlp.charlm.flair_converter.go` to
  import [Flair character language models](https://github.com/flairNLP/flair/issues/614).

### Changed

- Improve `nlp.transformer.generation` algorithms:
  - optimize `Generator.getTopKScoredTokens()`.
  - optimize `Generator.updateTokensScores()`.
- Simplify `mat32.Dense.Mul` when doing Matrix-Vector multiplication.
- Refactor `math32` functions using [chewxy/math32](https://github.com/chewxy/math32) functions.
- Improve `ag.Graph` efficiency:
  - Use pre-computed cache doing `ag.Graph.groupNodesByHeight()`.
  - Use `sync.pool` to reduce allocations of graph's operators.

### Fixed

- Fix past key-values usage on self-attention and cross-attention

## [0.5.0] - 2021-02-15

### Added

- Implement a beam-search algorithm for conditional generation:
  - `nlp.transformer.generation` package.
- Add implementation of the Sentence-Piece tokenizer:
  - `nlp.tokenizers.sentencepiece` package.
- BART improvements:
  - gRPC and HTTP API to perform Text Generation.
  - Add support for "Marian" architecture (used for translation tasks).
  - Add sinusoidal positional encoder (used by Marian).
  - Add "head" for conditional generation:
    - `nlp.transformers.bart.head.conditionalgeneration` package.
- Add `nn.Closer` interface (e.g. `embeddings.Model` needs to close the underlying key-value store).
- Add Swish act. function without trainable parameters.
- Add SiLU act. function (it is just an alias for Swish).
- New `pe.SinusoidalPositionalEncoder` (this implementation replaces unused `pe.PositionalEncoder`
  and `pe.AxialPositionalEncoder`)

### Changed

- Update urfave/cli to v2
- Update dgraph-io/badger to v3.
- Make the BART positional encoder an interface to support various encoding (i.e. trainable vs static).
- Rename to `fn.NewSwish` into `fn.NewSwishB` as this was the Swish variant with trainable parameters (*B*).
- Relax `ag.GetOpName` to match operator names in lower-case.
- Allow arbitrary activation function on BART encoder/decoder layers.
- Use precomputed "keys" and "values" in self-attention, multi-head attention and BART decoder.

### Removed

- In relation to the aforementioned positional encoding changes:
  - `pe.PositionalEncoder` and related functions
  - `pe.AxialPositionalEncoder` and related functions

### Fixed

- Fix causal-mask used by `nn.ScaledDotProductAttention`

## [0.4.1] - 2021-01-22

### Added

- New function `ReleaseMatrix` to packages `mat32` and `mat64`.
- New methods to `Matrix` interface, from `mat32` and `mat64`: `Minimum`,
  `Maximum`, `MulT`, `Inverse`, `DoNonZero`. However, the implementation on sparse matrices is not implemented yet (it
  always panics).

### Changed

- Prefer handling `Matrix` interface values over specific `Dense` or `Sparse`
  matrices, also avoiding unnecessary type casts. Relevant changes to the public API are listed below.
  - `mat(32|64).Stack` function's arguments and returned value are now `Matrix` 
    interfaces, instead of explicit `Dense` matrices.
  - `Dense.Minimum` and `Dense.Maximum`, from packages `mat32` and `mat64`,
    return a `Matrix` interface, instead of a specific `Dense` type.
  - The return values of `fofe.EncodeDense`, `fofe.Encode`, and `fofe.BiEncode`
    are slices of `Matrix` values, instead of `Dense` or `Sparse`.
  - The `z` argument of the function `fofe.Decode` is of type `Matrix`,
    instead of `Dense`.
  - `ml.optimizers.de` (Differential Evolution optimizer) API was changed
    handling `Matrix` values, instead of specific `Dense` matrices. Changes
    include: `Member.TargetVector`, `Member.DonorVector`, `ScoredVector.Vector`, 
    the `vector` argument of `NewMember` function, the `solution` argument
    of `score` and `validate` functions passed to `NewOptimizer`.
  - `PositionalEncoder.Cache` and `AxialPositionalEncoder.Cache` are slices
    of `Matrix`, instead of slices of `Dense`.
  - `AxialPositionalEncoder.EncodingAt` returns a `Matrix` value, instead of `Dense`.
  - `nn.DumpParamsVector` returns a `Matrix` value, instead of `Dense`.
  - The `vector` argument of the function `nn.LoadParamsVector` is a `Matrix`, 
    instead of `Dense`.
  - The `value` argument of the method `embeddings.Model.SetEmbedding` is of
    type `Matrix`, instead of `Dense`.
  - The type of the struct field `evolvingembeddings.WordVectorPair.Vector` is
   `Matrix`, instead of `Dense`.

## [0.4.0] - 2021-01-17

### Added

- Various new test cases (improving the coverage).
- `nlp.embeddings.syncmap` package.
- `ml.nn.recurrent.srnn.BiModel` which implements a bidirectional variant of the Shuffling Recurrent Neural Networks (
  SRNN).
- Configurable timeout and request limit to all HTTP and gRPC servers (see also commands help).

### Changed

- All CLI commands implementation has been refactored, so that the
  `docker-entrypoint` can reuse all other `cli.App` objects, instead of
  just running separate executables. By extension, now the Dockerfile builds
  a single executable file, and the final image is way smaller.
- All dependencies have been upgraded to the latest version.
- Simplify custom error definitions using `fmt.Errorf` instead of functions
  from `github.com/pkg/errors`.
- Custom binary data serialization of matrices and models is now achieved
  with Go's `encoding.gob`. Many specific functions and methods are now
  replaced by fewer and simpler encoding/decoding methods compatible with
  `gob`. A list of important related changes follows.
  - `utils.kvdb.KeyValueDB` is no longer an interface, but a struct which
    directly implements the former "badger backend".
  - `utils.SerializeToFile` and `utils.DeserializeFromFile` now handle
    generic `interface{}` objects, instead of values implementing
    `Serializer` and `Deserializer`.
  - `mat32` and `mat64` custom serialization functions (e.g.
    `MarshalBinarySlice`, `MarshalBinaryTo`, ...) are replaced by
    implementations of `BinaryMarshaler` and `BinaryUnmarshaler` interfaces
    on `Dense` and `Sparse` matrix types.
  - `PositionalEncoder.Cache` and `AxialPositionalEncoder.Cache` fields (from
    `ml.encoding.pe` package) are now public.
  - All types implementing `nn.Model` interface are registered for gob
    serialization (in init functions).
  - `embeddings.Model.UsedEmbeddings` type is now `nlp.embeddings.syncmap.Map`.
  - As a consequence, you will have to re-serialize all your models.
- Flair converter now sets the vocabulary directly in the model, instead
  of creating a separate file.
- `sequencelabeler.Model.LoadParams` has been renamed to `Load`.

### Removed
- In relation to the aforementioned gob serialization changes:
  - `nn.ParamSerializer` and related functions
  - `nn.ParamsSerializer` and related functions
  - `utils.Serializer` and `utils.Deserializer` interfaces
  - `utils.ReadFull` function
- `sequencelabeler.Model.LoadVocabulary`

### Fixed
- `docker-entrypoint` sub-command `hugging-face-importer` has been renamed to
  `huggingface-importer`, just like the main command itself.
- `docker-entrypoint` sub-command can be correctly specified without leading
  `./` or `/` when run from a Docker container.
- BREAKING: mat32.Matrix serialization has been fixed, now serializing single
  values to chunks of 4 bytes (instead of 8, like float64). Serialized 32-bit
  models will now be half the size! Unfortunately you will have to re-serialize
  your models (sorry!).

## [0.3.0] - 2021-01-10
### Added
- Static analysis job (`golint` and `gocyclo`) to `Go` GitHub workflow.
- You can set a limit for concurrent heavyweight Graph computations (e.g.
  forward and backward steps) - see `ml.ag.ConcurrentComputations()`
  (`GraphOption`) and `ml.ag.Graph.ConcurrentComputations()`.
  If no option is specified, by default the limit is set to `runtime.NumCPU()`.
- You can set a limit for concurrent heavyweight computations of
  `ml.optimizers.gd.GradientDescent` (e.g. params update step).
- New package `utils.processingqueue`.
- `mat32` package, which operates on `float32` data type.
- It's possible to switch between `float32` and `float64` as default
  floating-point data type, using the script `change-float-type.sh`
- `Go` GitHub workflow has been adapted to run tests using both `float32`
  and `float64` as main floating-point data type.
- This CHANGELOG file.
- Pull and convert Hugging Face models automatically if not found locally 
  when starting BERT or BART server.
- Move content from GitHub Wiki to README in related package folders.

### Changed
- `ml.ag.ConcurrentComputations` (`GraphOption`) expects the maximum number
  of concurrent computations handled by heavyweight Graph operations (e.g.
  forward and backward steps).
- `ml.nn.linear.Model` and `ml.nn.convolution.Model` read the concurrent
  computations limit set on the model's Graph, thus
  `SetConcurrentComputations()` methods have been removed.
- `mat` has been renamed to `mat64` and some functions have been renamed.
- The whole project now works with `float32` floating-point data type by
  default, by using the package `mat32`.
- When imported, the new package `mat32` is always aliased as `mat`. Then,
  explicit usages of `float64` type have been replaced with `mat.Float`.
  Moreover, bitsize-specific functions have been made more generic (i.e.
  operating with `mat.Float` type) or split into separate implementation,
  in `mat32` and `mat64`. In this way, switching the whole project between
  `float32` and `float64` is just a matter of changing all imports, from
  `mat32` to `mat64`, or vice-versa (see also the new file
  `change-float-type.sh`).
- Update internal links to pre-trained NER models to float32 versions.
- `nlp.sequencelabeler.Convert()` now loads and converts original Flair models,
   instead of pre-processed dumps.
- Change command line arguments to make them more consistent; please refer to
  the help messages of each command.
- Update Dockerfile using a new base building image and adding bart server.

### Fixed
- Added dedicated package names to different protocol buffers definition files
  to avoid name conflicts.

## [0.2.0] - 2020-12-31
### Added
- Support for BART model (tested on Natural Language Inference task).
- BART API to perform Zero-Shot Text Classification.

### Changed
- Significant reduction of boilerplate code through the unification of
  `nn.Model` and `nn.Processor` interfaces:
  - there is now a single `nn.Model` interface that can be reified to become a
    neural processor - see `nn.Reify()`;
  - there was no compelling reason to have a `Forward` method in the `nn.Model`
    interface, so it has been removed, gracefully increasing flexibility in the
    implementation of a model.

## [0.1.0] - 2020-12-09
First tagged release!
