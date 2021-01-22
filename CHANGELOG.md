# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
