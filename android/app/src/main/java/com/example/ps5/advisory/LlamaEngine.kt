package com.example.ps5.advisory

import android.os.SystemClock
import android.util.Log
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

data class GenerationResult(
    val text: String,
    val loadMs: Long,
    val ttftMs: Long,
    val tokens: Int,
    val totalMs: Long,
    val timedOut: Boolean,
    val error: String?,
    val statsJson: String?,
)

/**
 * Llama 3.2 1B Instruct on ExecuTorch (XNNPACK). Must only be called from one dedicated thread.
 * Files: <external files>/models/llama.pte and tokenizer.model (pushed with adb, see README).
 */
class LlamaEngine(private val modelDir: File) {
    val modelFile = File(modelDir, "llama.pte")
    val tokenizerFile = File(modelDir, "tokenizer.model")
    private var module: LlmModule? = null
    var lastLoadMs: Long = 0
        private set

    fun filesPresent() = modelFile.isFile && tokenizerFile.isFile
    fun isLoaded() = module != null

    @Synchronized
    fun ensureLoaded(): Long {
        if (module != null) return 0
        check(filesPresent()) { "Llama files missing in ${modelDir.absolutePath}" }
        val t0 = SystemClock.elapsedRealtime()
        val m = LlmModule(modelFile.absolutePath, tokenizerFile.absolutePath, TEMPERATURE)
        m.load() // throws on failure in ExecuTorch 1.5.1 (returns Unit)
        module = m
        lastLoadMs = SystemClock.elapsedRealtime() - t0
        return lastLoadMs
    }

    /**
     * [stopWhen] is checked on the accumulated text after every token; generation is stopped once it is true or
     * [maxNewTokens] tokens were produced (the runner's own limit was not honoured in the first device test:
     * up to 590 tokens with maxNewTokens = 110).
     */
    fun generate(
        prompt: String,
        maxNewTokens: Int,
        timeoutMs: Long,
        stopWhen: (String) -> Boolean = { false },
        onToken: (String) -> Unit = {},
    ): GenerationResult {
        val loadMs = ensureLoaded()
        val m = module!!
        m.resetContext() // every inspection is a fresh conversation
        val out = StringBuilder()
        var tokens = 0
        var first = -1L
        var error: String? = null
        var stats: String? = null
        val done = CountDownLatch(1)
        val t0 = SystemClock.elapsedRealtime()
        val config = LlmGenerationConfig.create()
            .maxNewTokens(maxNewTokens)
            .seqLen(SEQ_LEN)
            .temperature(TEMPERATURE)
            .numBos(0) // prompt already starts with <|begin_of_text|>
            .numEos(0)
            .echo(false)
            .build()
        var stopRequested = false
        val callback = object : LlmCallback {
            override fun onResult(result: String) {
                if (stopRequested) return
                if (first < 0) first = SystemClock.elapsedRealtime() - t0
                tokens++
                if (result == "<|eot_id|>" || result == "<|end_of_text|>") return
                out.append(result)
                onToken(result)
                if (tokens >= maxNewTokens || stopWhen(out.toString())) {
                    stopRequested = true
                    m.stop()
                }
            }
            override fun onStats(s: String) { stats = s }
            override fun onError(code: Int, message: String) { error = "$code: $message" }
        }
        val worker = Thread {
            try {
                m.generate(prompt, config, callback)
            } catch (t: Throwable) {
                error = t.message ?: t.javaClass.simpleName
            } finally {
                done.countDown()
            }
        }
        worker.start()
        val finished = done.await(timeoutMs, TimeUnit.MILLISECONDS)
        if (!finished) {
            m.stop()
            done.await(5, TimeUnit.SECONDS)
        }
        val total = SystemClock.elapsedRealtime() - t0
        Log.i(TAG, "generated $tokens tokens in $total ms (ttft $first ms, timeout=${!finished})")
        return GenerationResult(out.toString(), loadMs, first, tokens, total, !finished, error, stats)
    }

    @Synchronized
    fun unload() {
        module?.close()
        module = null
    }

    companion object {
        const val TAG = "LlamaEngine"
        const val TEMPERATURE = 0.2f
        const val SEQ_LEN = 768
        const val MAX_NEW_TOKENS = 110
    }
}
