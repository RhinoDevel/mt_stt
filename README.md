# mt_stt

*Marcel Timm, RhinoDevel, 2025*

**mt_stt** is a C++ library for Linux and Windows that offers a pure C interface
to the great speech-to-text inference engine
[Whisper.cpp](https://github.com/ggml-org/whisper.cpp) by Georgi Gerganov that
itself runs [OpenAI Whisper](https://github.com/openai/whisper) models.

With **mt_stt** you can:
- Transcribe from raw audio in memory to a string.
- Use a model to be loaded from file or already held in memory.
- Translate to English.
- Add an optional initial prompt (to bias/help the transcription process).
- Progress callback and cancel option.
- Optionally transcribe a specific part of the audio data, only.
- Output probabilities of the transcribed words (how sure the model is about the
  word representing the correct result).

## How To

After cloning the **mt_stt** repository, enter its folder and proceed as
follows.

Get the [Whisper.cpp](https://github.com/ggml-org/whisper.cpp) submodule
content:

`git submodule update --init --recursive`

## Linux

No details for Linux here, yet, but you can take a look at the Windows
instructions below and at the [Makefile](./mt_stt/Makefile).

## Windows

All the following examples are building static libraries, there may be use cases
where dynamically linked libraries are sufficient, too.

### Build [Whisper.cpp](https://github.com/ggml-org/whisper.cpp)

#### Compile `whisper.lib` and `ggml.lib` as static libraries

Compile the necessary `whisper.lib` and `ggml.lib` libraries via Visual Studio
and `mt_stt/whisper.cpp/CMakeLists.txt` as static libraries.

To do that, modify the file `mt_stt/whisper.cpp/CMakePresets.json` which is
created by Visual Studio:

If the binary of `git` is not in your path, modify `"configurePresets"` entry
with `"name"` `"windows-base"` by adding the following entry to
`"cacheVariables"`:

`"GIT_EXE": "C:\\Program Files\\Git\\bin\\git.exe"`

Add entry

```
{
  "name": "mt-x64-release-static",
  "displayName": "MT x64 Release Static (native)",
  "description": "MT: Target Windows (64-bit), static, with the Visual Studio development environment. (RelWithDebInfo)",
  "inherits": "x64-release",
  "cacheVariables": {
    "BUILD_SHARED_LIBS": "OFF"
  }
}
```

to `mt_stt/whisper.cpp/CMakePresets.json`'s `configurePresets` array.

#### OpenBLAS build

Download [OpenBLAS](http://www.openmathlib.org/OpenBLAS/) (e.g.
`OpenBLAS-0.3.29-x64.zip`) and unpack the content to `C:\openblas`.

Additionally add entry

```
{
  "name": "mt-x64-release-static-blas",
  "displayName": "MT x64 Release Static BLAS",
  "description": "MT: Target Windows (64-bit), static, BLAS, with the Visual Studio development environment. (RelWithDebInfo)",
  "inherits": "mt-x64-release-static",
  "cacheVariables": {
    "GGML_BLAS": "ON",
    "BLAS_LIBRARIES": "C:/openblas/lib/libopenblas.lib",
    "BLAS_INCLUDE_DIRS": "C:/openblas/include"
  }
}
```

Put the `libopenblas.dll` (from `C:\openblas\bin\libopenblas.dll`) into the
folder of the executable file that will be linked with THIS project's resulting
DLL.

#### CUDA build

Working with (e.g.): CUDA 12.4.131 and Whisper.cpp v1.7.5

Additionally add entry

```
{
  "name": "mt-x64-release-static-cuda",
  "displayName": "MT x64 Release Static CUDA (native)",
  "description": "MT: Target Windows (64-bit), static, CUDA, with the Visual Studio development environment. (RelWithDebInfo)",
  "inherits": "mt-x64-release-static",
  "cacheVariables": {
    "GGML_CUDA": "ON"
  }
}
```

to `mt_stt/whisper.cpp/CMakePresets.json`'s configurePresets array.

In **mt_stt**, link with these libraries (e.g. from `C:\cuda\lib\`):

- `x64\cublas.lib`
- `x64\cuda.lib`
- `x64\cudart.lib`

Put the following files (e.g. from `C:\cuda\bin`) into the folder of the
executable file that will be linked with **this** project's resulting DLL:

- `cublas64_12.dll`
- `cublasLt64_12.dll`
- `cudart64_12.dll`

On a non-development PC, make sure that the most recent Nvidia drivers are
installed (they include CUDA support).

#### Build for non-AVX processors (e.g. Celeron)

Additionally add entry

```
{
  "name": "mt-x64-release-static-sse",
  "displayName": "MT x64 Release Static SSE",
  "description": "MT: Target Windows (64-bit), static, SSE, with the Visual Studio development environment. (RelWithDebInfo)",
  "inherits": "mt-x64-release-static",
  "cacheVariables": {
    "GGML_NATIVE": "OFF",
    "GGML_AVX": "OFF",
    "GGML_AVX2": "OFF"
  }
}
```

to `mt_stt/whisper.cpp/CMakePresets.json`'s configurePresets array.

**and** change the line

`#if defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))`

to

`#if defined(_MSC_VER)// && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))`

in the file

`mt_stt/whisper.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h`

before the line

`#ifndef __SSE3__`

to enable SSE3 and SSSE3.

#### Build for non-AVX processors (e.g. Celeron), with OpenBLAS

Download [OpenBLAS](http://www.openmathlib.org/OpenBLAS/) (e.g.
`OpenBLAS-0.3.29-x64.zip`) and unpack the content to `C:\openblas`.

Additionally add entry (also don't forget `ggml-cpu-impl.h` - see above)

```
{
  "name": "mt-x64-release-static-sse-blas",
  "displayName": "MT x64 Release Static SSE and BLAS",
  "description": "MT: Target Windows (64-bit), static, SSE, BLAS, with the Visual Studio development environment. (RelWithDebInfo)",
  "inherits": "mt-x64-release-static-sse",
  "cacheVariables": {
    "GGML_BLAS": "ON",
    "BLAS_LIBRARIES": "C:/openblas/lib/libopenblas.lib",
    "BLAS_INCLUDE_DIRS": "C:/openblas/include"
  }
}
```

Put the `libopenblas.dll` (from `C:\openblas\bin\libopenblas.dll`) into the folder
of the executable file that will be linked with **this** project's resulting DLL.

### Build mt_stt

- Open solution `mt_stt.sln` with Visual Studio (tested with 2022).
- Compile in release or debug mode.

### Test mt_stt

- The sample code below is using [mt_tts](https://github.com/RhinoDevel/mt_tts),
  which is kind of the counterpart to **this** project.
- Follow [Test mt_tts](https://github.com/RhinoDevel/mt_tts?tab=readme-ov-file#test-mt_tts)
  first.
- Get the DLL and LIB files resulting from building **this** project, e.g. for
  release mode `x64\Release\mt_stt.dll` and `x64\Release\mt_stt.lib`, copy them
  to the folder from [Test mt_tts](https://github.com/RhinoDevel/mt_tts?tab=readme-ov-file#test-mt_tts).
- Also copy the file `mt_stt\mt_stt.h` to that folder.
- Copy a [Whisper(.cpp) model file](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin) that supports translation to English to the same new folder.
- Open `x64 Native Tools Command Prompt for VS 2022` commandline.
- Go to the example folder and put the following code into the already existing file `main.c`:

```
#include <stdio.h>
#include <stdlib.h>

#include "mt_tts.h"
#include "mt_stt.h"

/** Example use of mt_stt transcribing & translating German language audio to
 *  text in English.
 *
 *  The audio is generated first with mt_tts.
 */
int main(void)
{
    int16_t* tts_result = NULL;
    int sample_count = -1;
    float* stt_input = NULL;
    char* stt_result = NULL;

    // *************************************************************************
    // *** TTS: Create raw audio data from a text given in German:           ***
    // *************************************************************************

    // Initialize TTS system with a model/voice for output in German:
    mt_tts_reinit("de_DE-thorsten-high.onnx", "de_DE-thorsten-high.onnx.json");

    // Get the actual raw audio data:
    tts_result = mt_tts_to_raw(
        "Hallo! Dies ist ein Text in deutscher Sprache. Erst wird er in ein Tonsignal umgewandelt, welches dann wiederum in Text transkribiert wird, jedoch nun auf Englisch.",
        &sample_count);

    // Convert the audio data into normalized floating-point representation:

    stt_input = malloc(sample_count * sizeof *stt_input);

    for(int i = 0; i < sample_count; ++i)
    {
        stt_input[i] = (float)tts_result[i] / 16384.0f;
    }

    // Free memory and de-initialize TTS system:

    mt_tts_free_raw(tts_result);
    tts_result = NULL;

    mt_tts_deinit();
    
    // *************************************************************************
    // *** STT: Transcribe the audio while also translating it to English:   ***
    // *************************************************************************
    
    stt_result = mt_stt_transcribe_with_file(
        false,
        4,
        NULL,
        true,
        NULL,
        "ggml-small-q5_1.bin",
        stt_input,
        sample_count,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        0);

    // Output the translated transcription of the spoken text:
    printf("%s\n", stt_result);

    // Free memory and exit:
    free(stt_result);
    stt_result = NULL;
    return 0;
}
```

- Compile via `cl main.c mt_tts.lib mt_stt.lib`.
- Run `main.exe`, which should show the transcription/translation result.

### Notes

- Install Microsoft Visual C++ Redistributable Version for Visual Studio 2015,
  2017, 2019, and 2022 (e.g. version 14.42.34433.0).