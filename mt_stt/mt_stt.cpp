
// Marcel Timm, RhinoDevel, 2024aug04

// This is kind of a hack:
//
#ifndef MT_EXPORT_STT
    #define MT_EXPORT_STT
#endif //MT_EXPORT_STT

#include "mt_stt.h"
#include "whisper.h"

#include <cassert>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

static char const * const s_log_file_path = "mt_stt_log.txt";

// These are all to be initialized and de-initialized via mt_stt_transcribe():
//
static FILE* s_log_file = nullptr;
static void (*s_on_progress_func)(int progress) = nullptr;
static bool s_aborted = false; // To be set to true by mt_stt_cancel(), only.
static int s_parts_index = -1;
static int s_parts_count = -1;

/**
 * - Caller takes ownership of return value.
 */
static char* create_copy(std::string const & str_ref)
{
    char* ret_val = nullptr;
    size_t const bytes = (str_ref.length() + 1) * sizeof *ret_val;

    ret_val = (char*)malloc(bytes);
    if(ret_val == nullptr)
    {
        return nullptr; // Must not get here.
    }

    // Ignoring return value:
    //
#ifdef _WIN32
    strcpy_s(ret_val, static_cast<rsize_t>(bytes), str_ref.c_str());
#else //_WIN32
    strcpy(ret_val, str_ref.c_str());
#endif //_WIN32

    return ret_val;
}

// This works, but is currently unused:
// 
//static std::string create_left_trimmed(std::string const & str_ref)
//{
//    size_t start = 0;
//
//    while(start < str_ref.size()
//        && std::isspace(static_cast<unsigned char>(str_ref[start])))
//    {
//        ++start;
//    }
//    return str_ref.substr(start);
//}

static void on_log(ggml_log_level level, const char * text, void* user_data)
{
    if(s_log_file == nullptr)
    {
        return;
    }

    fputs(text, s_log_file);
    fflush(s_log_file);
}

static void on_progress(
    struct whisper_context * ctx,
    struct whisper_state * state,
    int progress,
    void * user_data)
{
    assert(0 <= s_parts_index);
    assert(s_parts_index < s_parts_count);

    int full_progress = progress;

    if(s_on_progress_func == nullptr)
    {
        assert(false); // Should not get here.
        return;
    }

    if(1 < s_parts_count)
    {
        // E.g.:
        //
        // Index = 3
        // Count = 5
        // => 
        // Full progress = (100 * 3 + progress) / 5
        //
        full_progress = (100 * s_parts_index + progress) / s_parts_count;
    }

    //fprintf(s_log_file, "full_progress: %d\n", full_progress);

    s_on_progress_func(full_progress);
}

static bool on_is_abort(void * data)
{
    return s_aborted;
}
static bool on_encoder_begin(
    struct whisper_context * ctx,
    struct whisper_state * state,
    void * user_data)
{
    return !on_is_abort(nullptr);
}

/** Get the results from a transcription and optionally just ADD to the given
 *  word probabilities vector (that may not be empty, which is OK).
 */
static std::string get_result_as_text(
    struct whisper_context * const ctx, std::vector<float> * const word_probs)
{
    whisper_token const tok_eot = whisper_token_eot(ctx);

    std::string ret_val = "";

    for(int i = 0; i < whisper_full_n_segments(ctx); ++i)
    {
        if(word_probs == nullptr) // <=> No probabilites wanted.
        {
            ret_val += whisper_full_get_segment_text(ctx, i);
            continue;
        }

        // Caller wants the probability for each word.

        for(int j = 0; j < whisper_full_n_tokens(ctx, i); ++j)
        {
#ifndef NDEBUG
            fprintf(
                s_log_file,
                "%d;%d;\"%s\";%f;\n",
                i,
                j,
                whisper_full_get_token_text(ctx, i, j),
                whisper_full_get_token_p(ctx, i, j));
#endif //NDEBUG

            if(tok_eot <= whisper_full_get_token_id(ctx, i, j))
            {
                continue; // Skip this special token.
            }

            std::string const tok_text = whisper_full_get_token_text(ctx, i, j);

            ret_val += tok_text;

            // We want one probability per word. If a token starts with a
            // whitespace, it is interpreted as the beginning of a word, here:
            //
            assert(0 < tok_text.length());
            if(std::isspace(static_cast<unsigned char>(tok_text[0])))
            {
                // Just using the probability of the word's first token as
                // the (whole) word's probability:
                //
                word_probs->push_back(whisper_full_get_token_p(ctx, i, j));
            }
        }
    }
    return ret_val;
}

static char* transcribe(
    bool const use_gpu,
    int const n_threads,
    char const * const language,
    bool const translate_to_en,
    char const * const initial_prompt,
    char * const model_file_path,
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
    int const opt_parts_length)
{
    assert(
        (opt_out_word_probs == nullptr)
            == (opt_out_word_probs_count == nullptr));

    assert(
        (model_file_path == nullptr
            && model_data != nullptr && 0 < model_data_len)
        || (model_file_path != nullptr
                && model_data == nullptr && model_data_len == -1));

    assert(
        (opt_out_parts_ret_val_indices == nullptr
            && opt_parts_audio_data_indices == nullptr
            && opt_parts_audio_data_limits == nullptr
            && opt_parts_length == 0)
        || (opt_out_parts_ret_val_indices != nullptr
            && opt_parts_audio_data_indices != nullptr
            && opt_parts_audio_data_limits != nullptr
            && 0 < opt_parts_length));

    std::string buf;
    whisper_context_params ctx_p;
    struct whisper_full_params params;
    std::vector<float> word_probs;
    std::vector<whisper_token> prompt_tokens;

    whisper_log_set(on_log, NULL);
    assert(s_log_file == nullptr);

#ifdef _WIN32
    s_log_file = _fsopen(s_log_file_path, "a", SH_DENYWR);
#else //_WIN32
    s_log_file = fopen(s_log_file_path, "a");
#endif //_WIN32
    if(s_log_file == nullptr)
    {
        return nullptr;
    }
    assert(s_log_file != nullptr);

    // Print given parameters:
    //
//#ifndef NDEBUG
    fprintf(s_log_file, "use_gpu = %d\n", (int)use_gpu);
    fprintf(s_log_file, "n_threads = %d\n", n_threads);
    fprintf(
        s_log_file,
        "language = \"%s\"\n",
        language == NULL ? "(null)" : language);
    fprintf(s_log_file, "translate_to_en = %d\n", (int)translate_to_en);
    fprintf(
        s_log_file,
        "initial_prompt = \"%s\"\n",
        initial_prompt == NULL ? "(null)" : initial_prompt);
//#endif //NDEBUG

    ctx_p = whisper_context_default_params();
    ctx_p.use_gpu = use_gpu;

    struct whisper_context * const ctx = model_file_path == nullptr
            ? whisper_init_from_buffer_with_params(
                model_data, model_data_len, ctx_p)
            : whisper_init_from_file_with_params(model_file_path, ctx_p);
    
    params = whisper_full_default_params(
       WHISPER_SAMPLING_GREEDY);
       //WHISPER_SAMPLING_BEAM_SEARCH); // Does not seem to do any magic.

    if(0 < n_threads)
    {
        params.n_threads = n_threads;
    }
    //
    // Otherwise: Will be set based on hardware.

    // Manually creating tokens from initial prompt (if given), to be able to
    // abort, if initial prompt is too long [see whisper_full_with_state()]:
    //
    if(initial_prompt != nullptr && 0 < strlen(initial_prompt))
    {
        assert(params.prompt_tokens == nullptr);
        assert(params.prompt_n_tokens == 0);

        int n_needed = 0;

        prompt_tokens.resize(1024);
        
        n_needed = whisper_tokenize(
            ctx,
            initial_prompt,
            prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size()));
        if(n_needed < 0)
        {
            prompt_tokens.resize(-n_needed);

            n_needed = whisper_tokenize(
                ctx,
                initial_prompt,
                prompt_tokens.data(),
                static_cast<int>(prompt_tokens.size()));
        }

        int const max_initial_prompt_tokens = whisper_n_text_ctx(ctx) / 2;

#ifndef NDEBUG
        fprintf(
            s_log_file,
            "max_initial_prompt_tokens: %d; n_needed: %d\n",
            max_initial_prompt_tokens,
            n_needed);
#endif //NDEBUG

        if(max_initial_prompt_tokens < n_needed)
        {
            fprintf(
                s_log_file,
                "Error: Initial prompt is too long (%d tokens, max. is %d tokens)!\n",
                n_needed,
                max_initial_prompt_tokens);
            whisper_free(ctx);
            fclose(s_log_file);
            s_log_file = nullptr;
            return nullptr;
        }

        prompt_tokens.resize(n_needed);
        
        params.initial_prompt = initial_prompt; // Probably not necessary.
        params.prompt_tokens = prompt_tokens.data();
        params.prompt_n_tokens = static_cast<int>(prompt_tokens.size());
    }

    params.translate = translate_to_en;
    params.no_context = false; // Keep context between parts (if given).
    params.language = language;
    params.detect_language = false; // This leads to "just" detecting the language, as it seems.
    params.suppress_blank = true;
    params.suppress_nst = true;
    //params.no_timestamps = false/*true*/; // Must be implemented manually.
    //params.print_special = true/*false*/; // Must be implemented manually.
    //params.print_progress = true/*false*/;

    assert(s_on_progress_func == nullptr);
    if(on_progress_func != nullptr)
    {
        s_on_progress_func = on_progress_func;

        params.progress_callback = on_progress;
        //params.progress_callback_user_data;
    }
    assert(s_parts_count == -1);
    s_parts_count = opt_parts_length != 0 ? opt_parts_length : 1;

    params.abort_callback = on_is_abort;
    params.abort_callback_user_data = nullptr;
    //
    params.encoder_begin_callback = on_encoder_begin;
    params.encoder_begin_callback_user_data = nullptr;

    fprintf(s_log_file, "%s\n", whisper_print_system_info());

    whisper_reset_timings(ctx);

    bool const get_word_probs = opt_out_word_probs != nullptr;

    assert(s_parts_index == -1);
    if(opt_out_parts_ret_val_indices == nullptr) // => One single "part".
    {
        s_parts_index = 0;

        if(whisper_full(ctx, params, audio_data_arr, audio_data_length) != 0)
        {
            whisper_free(ctx);
            s_on_progress_func = nullptr;
            s_parts_count = -1;
            s_parts_index = -1;
            fclose(s_log_file);
            s_log_file = nullptr;
            return nullptr;
        }
        buf = get_result_as_text(ctx, get_word_probs ? &word_probs : nullptr);
    }
    else // => Transcribe given parts of the audio data, only.
    {
        buf = "";
        for(int i = 0; i < opt_parts_length; ++i) // Transcribe each given part.
        {
            s_parts_index = i;

            // 0 1 2 3 4 5 6 7 8 9
            //     ^             ^
            //     |             |
            //     index         limit
            // 
            // length = limit - index = 9 - 2 = 7

            int const part_audio_index = opt_parts_audio_data_indices[i];
            float const * part_audio_data =
                    audio_data_arr + part_audio_index;
            int part_audio_data_length =
                    opt_parts_audio_data_limits[i] - part_audio_index;

            // Pad audio data, if less than a second (necessary for Whisper):
            //
            // * Hard-coded for a sample rate of 16000 Hz!
            //            
            static int const min_audio_data_len = 16000 + 384;
            float* min_buf = nullptr;
            //
            if(part_audio_data_length < min_audio_data_len)
            {
                min_buf = (float*)malloc(min_audio_data_len * sizeof * min_buf);

                assert(min_buf != nullptr);

                for(int j = 0; j < min_audio_data_len; ++j)
                {
                    min_buf[j] = 0.0f;

                    if(j < part_audio_index)
                    {
                        min_buf[j] = part_audio_data[j];
                    }
                }
                part_audio_data = min_buf;
                part_audio_data_length = min_audio_data_len;
            }

            if(whisper_full(
                ctx, params, part_audio_data, part_audio_data_length)
                    != 0)
            {
                part_audio_data = nullptr;
                part_audio_data_length = 0;
                if(min_buf != nullptr)
                {
                    free(min_buf);
                    min_buf = nullptr;
                }

                whisper_free(ctx);
                s_on_progress_func = nullptr;
                s_parts_count = -1;
                s_parts_index = -1;
                fclose(s_log_file);
                s_log_file = nullptr;
                return nullptr;
            }

            part_audio_data = nullptr;
            part_audio_data_length = 0;
            if(min_buf != nullptr)
            {
                free(min_buf);
                min_buf = nullptr;
            }

            std::string const cur_text = get_result_as_text(
                ctx, get_word_probs ? &word_probs : nullptr);

            //fprintf(
            //    s_log_file,
            //    "CUR_TEXT AT %d: \"%s\"\n",
            //    (int)buf.length(),
            //    cur_text.c_str());

            opt_out_parts_ret_val_indices[i] = -1;
            if(cur_text.length() != 0)
            {
                opt_out_parts_ret_val_indices[i] = (int)buf.length();

                buf += cur_text;
            }
        }
    }

    whisper_print_timings(ctx);
    
    if(get_word_probs)
    {
        *opt_out_word_probs = nullptr;
        *opt_out_word_probs_count = 0;
        if(!word_probs.empty())
        {
            size_t const bytes =
                word_probs.size() * sizeof **opt_out_word_probs;

            *opt_out_word_probs = (float*)malloc(bytes);
            if(*opt_out_word_probs == nullptr)
            {
                whisper_free(ctx);
                s_on_progress_func = nullptr;
                s_parts_count = -1;
                s_parts_index = -1;
                fclose(s_log_file);
                s_log_file = nullptr;
                return nullptr; // Must not get here.
            }

            *opt_out_word_probs_count = (int)word_probs.size();
            for(int i = 0; i < *opt_out_word_probs_count; ++i)
            {
                *((*opt_out_word_probs) + i) = word_probs[i];
            }
        }
    }

    //fprintf(
    //    s_log_file,
    //    "CONTENT OF buf BEFORE RETURN: \"%s\"\n",
    //    buf.c_str());

    whisper_free(ctx);
    s_on_progress_func = nullptr;
    s_parts_count = -1;
    s_parts_index = -1;
    fclose(s_log_file);
    s_log_file = nullptr;

    return create_copy(buf);
}

MT_EXPORT_STT_API void __stdcall mt_stt_free(void * const ptr)
{
    free(ptr);
}

MT_EXPORT_STT_API void __stdcall mt_stt_cancel()
{
    s_aborted = true;
}

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
    int const opt_parts_length)
{
    return transcribe(
        use_gpu,
        n_threads,
        language,
        translate_to_en,
        initial_prompt,
        model_file_path,
        nullptr,
        -1,
        audio_data_arr,
        audio_data_length,
        on_progress_func,
        opt_out_word_probs,
        opt_out_word_probs_count,
        opt_out_parts_ret_val_indices,
        opt_parts_audio_data_indices,
        opt_parts_audio_data_limits,
        opt_parts_length);
}

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
    int const opt_parts_length)
{
    return transcribe(
        use_gpu,
        n_threads,
        language,
        translate_to_en,
        initial_prompt,
        nullptr,
        model_data,
        model_data_len,
        audio_data_arr,
        audio_data_length,
        on_progress_func,
        opt_out_word_probs,
        opt_out_word_probs_count,
        opt_out_parts_ret_val_indices,
        opt_parts_audio_data_indices,
        opt_parts_audio_data_limits,
        opt_parts_length);
}
