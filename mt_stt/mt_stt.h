
// Marcel Timm, RhinoDevel, 2024aug04

#ifndef MT_STT
#define MT_STT

#ifdef _WIN32
    #ifdef MT_EXPORT_STT
        #define MT_EXPORT_STT_API __declspec(dllexport)
    #else
        #define MT_EXPORT_STT_API __declspec(dllimport)
    #endif //MT_EXPORT_STT
#else //_WIN32
    #define MT_EXPORT_STT_API
    #ifndef __stdcall
        #define __stdcall
    #endif //__stdcall
#endif //_WIN32

// This is necessary to avoid function name mangling (even, if this is no pure C
// code):
//
#ifdef __cplusplus

#include <cstdbool>

extern "C" {

#else //__cplusplus

#include <stdbool.h>

#endif //__cplusplus

MT_EXPORT_STT_API void __stdcall mt_stt_free(void * const ptr);

MT_EXPORT_STT_API void __stdcall mt_stt_cancel();

/**
 * - Caller takes ownership of the returned, zero-terminated C-string.
 * - If opt_out_word_probs is not NULL, it will be set to a dynamically
 *   allocated array of word probabilities the caller takes ownership of and the
 *   array needs to be freed via mt_stt_free().
 * - If opt_out_parts_ret_val_indices is not NULL, only the parts of the
 *   input data identified by opt_parts_audio_data_indices (beginnings) and
 *   opt_parts_audio_data_limits (exclusive indices of ends) are transcribed and
 *   opt_out_parts_ret_val_indices will hold the indices of the first transcribed
 *   character in the return value for each (given and transcribed) part.
 *   opt_parts_length holds the count of the parts.
 * - Progress callback given is optional.
 * - Returns NULL on error.
 */
MT_EXPORT_STT_API char* __stdcall mt_stt_transcribe_with_file(
    bool const use_gpu,
    int const n_threads,
    char const * const language,
    bool const translate_to_en,
    char const * const initial_prompt,
    char * const model_file_path,
    float const * const audio_data_arr,
    int const audio_data_length,
    void (*on_progress_func)(int progress),
    float * * const opt_out_word_probs,
    int * opt_out_word_probs_count,
    int * const opt_out_parts_ret_val_indices,
    int const * const opt_parts_audio_data_indices,
    int const * const opt_parts_audio_data_limits,
    int const opt_parts_length);

/**
 * - Caller takes ownership of the returned, zero-terminated C-string.
 * - If opt_out_word_probs is not NULL, it will be set to a dynamically
 *   allocated array of word probabilities the caller takes ownership of and the
 *   array needs to be freed via mt_stt_free().
 * - If opt_out_parts_ret_val_indices is not NULL, only the parts of the
 *   input data identified by opt_parts_audio_data_indices (beginnings) and
 *   opt_parts_audio_data_limits (exclusive indices of ends) are transcribed and
 *   opt_out_parts_ret_val_indices will hold the indices of the first transcribed
 *   character in the return value for each (given and transcribed) part.
 *   opt_parts_length holds the count of the parts.
 * - Progress callback given is optional.
 * - Returns NULL on error.
 */
MT_EXPORT_STT_API char* __stdcall mt_stt_transcribe_with_data(
    bool const use_gpu,
    int const n_threads,
    char const * const language,
    bool const translate_to_en,
    char const * const initial_prompt,
    void * const model_data,
    size_t const model_data_len,
    float const * const audio_data_arr,
    int const audio_data_length,
    void (*on_progress_func)(int progress),
    float * * const opt_out_word_probs,
    int * opt_out_word_probs_count,
    int * const opt_out_parts_ret_val_indices,
    int const * const opt_parts_audio_data_indices,
    int const * const opt_parts_audio_data_limits,
    int const opt_parts_length);

#ifdef __cplusplus
}
#endif

#endif //MT_STT
