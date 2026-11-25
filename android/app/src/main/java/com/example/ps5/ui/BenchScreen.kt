package com.example.ps5.ui

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.ps5.AppEngine
import com.example.ps5.advisory.AdvisoryRecord
import com.example.ps5.advisory.AdvisoryValidator
import com.example.ps5.advisory.LlamaEngine
import com.example.ps5.advisory.PromptBuilder
import com.example.ps5.data.Csv
import com.example.ps5.vision.ItemFinder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.random.Random

fun deviceHeader(context: Context): String {
    val bm = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    val level = bm?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
    val temp = (bm?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0
    val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
    val mi = android.app.ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
    val airplane = android.provider.Settings.Global.getInt(context.contentResolver, android.provider.Settings.Global.AIRPLANE_MODE_ON, 0) == 1
    return "# device=${Build.MANUFACTURER} ${Build.MODEL}; android=${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}); ram_mb=${mi.totalMem / 1048576}; " +
        "battery_pct=$level; battery_temp_c=$temp; thermal_status=${pm.currentThermalStatus}; airplane=$airplane; ts=${stamp()}"
}

fun pssMb(): Double = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }.totalPss / 1024.0

fun summary(values: List<Long>): String {
    if (values.isEmpty()) return "n=0"
    val s = values.sorted()
    val med = s[s.size / 2]
    val p95 = s[((s.size - 1) * 0.95).toInt()]
    return "median=$med ms p95=$p95 ms n=${s.size}"
}

/**
 * Deterministic advisory test cases: flagged items only, all verdict/type/severity combinations.
 * Seed 2026 was used to diagnose prompt v1; prompt v2 is reported on the unseen seed 2027.
 */
const val ADVISORY_SEED = 2027

fun advisoryCases(n: Int, seed: Int = ADVISORY_SEED): List<AdvisoryRecord> {
    val rnd = Random(seed)
    val types = listOf("crack or scratch", "contamination or colour fault", "deformation", "unspecified anomaly")
    val locs = listOf("upper-left", "top", "upper-right", "left", "centre", "right", "lower-left", "bottom", "lower-right")
    return (1..n).map { i ->
        val verdict = if (i % 2 == 0) "REJECT" else "RE-INSPECT"
        val sev = listOf("LOW", "MEDIUM", "HIGH")[if (verdict == "REJECT") rnd.nextInt(3) else rnd.nextInt(2)]
        AdvisoryRecord(
            product = "pill", itemNo = 1 + rnd.nextInt(9), verdict = verdict, defectType = types[rnd.nextInt(types.size)],
            severity = sev, confidencePct = 50 + rnd.nextInt(50), areaPct = (rnd.nextInt(5, 400)) / 100.0,
            location = locs[rnd.nextInt(locs.size)], box = null, recentRejectsLast20 = rnd.nextInt(6),
        )
    }
}

@Composable
fun BenchScreen(engine: AppEngine, modifier: Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var log by remember { mutableStateOf("") }
    var running by remember { mutableStateOf(false) }
    fun say(s: String) { log = s + "\n" + log }

    fun job(name: String, block: suspend () -> Unit) {
        running = true
        scope.launch {
            try { block() } catch (t: Throwable) { say("$name failed: ${t.message}") } finally { running = false }
        }
    }

    Column(modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Bank in use: ${engine.bankSize}. Results in ${engine.resultsDir.path}", style = MaterialTheme.typography.bodySmall)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            AppEngine.BANKS.forEach { b ->
                OutlinedButton(enabled = !running, onClick = { job("load") { withContext(engine.visionDispatcher) { engine.loadVision(b) }; say("Loaded bank $b ${engine.visionError ?: ""}") } }) { Text("Use $b") }
            }
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Button(enabled = !running, onClick = { job("vision") { benchVision(engine, context, ::say) } }) { Text("Vision latency (all banks)") }
            Button(enabled = !running, onClick = { job("golden") { golden(engine, ::say) } }) { Text("Golden parity") }
            Button(enabled = !running, onClick = { job("fps") { fps(engine, context, ::say) } }) { Text("FPS 60 s") }
            Button(enabled = !running, onClick = { job("memory") { say("PSS now: %.1f MB (llama loaded=${engine.llama.isLoaded()})".format(pssMb())) } }) { Text("Memory") }
            Button(enabled = !running && engine.llama.filesPresent(), onClick = { job("llama") { llamaBench(engine, context, 50, ::say) } }) { Text("Advisory test (50 cases)") }
            Button(enabled = !running, onClick = { job("long") { longRun(engine, context, 15, ::say) } }) { Text("Long run 15 min") }
        }
        if (!engine.llama.filesPresent()) Text("Llama files not found in ${engine.modelDir.path}", color = MaterialTheme.colorScheme.error)
        Text(log, style = MaterialTheme.typography.bodySmall)
    }
}

private fun benchCrop(engine: AppEngine): Mat {
    val f = replayImages(engine.replayDir).firstOrNull()
    val rgb = Mat()
    if (f != null) {
        val rgba = Mat(); Utils.bitmapToMat(decode(f), rgba); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB); rgba.release()
    } else {
        Mat(800, 800, org.opencv.core.CvType.CV_8UC3, org.opencv.core.Scalar(15.0, 15.0, 26.0)).copyTo(rgb)
    }
    return ItemFinder.wholeImage256(rgb).also { rgb.release() }
}

private suspend fun benchVision(engine: AppEngine, context: Context, say: (String) -> Unit) {
    val original = engine.bankSize
    for (bank in AppEngine.BANKS) {
        withContext(engine.visionDispatcher) {
            engine.loadVision(bank)
            val crop = benchCrop(engine)
            repeat(20) { engine.pipeline.inspectCrop(crop, false) }
            val out = File(engine.resultsDir, "bench_vision_bank${bank}_${stamp()}.csv")
            val pre = mutableListOf<Long>(); val inf = mutableListOf<Long>(); val post = mutableListOf<Long>()
            out.printWriter().use { w ->
                w.println(deviceHeader(context) + "; threads=4; runtime=onnxruntime-android 1.23.2 CPU")
                w.println("iter,preprocess_ms,inference_ms,postprocess_ms,total_ms")
                repeat(200) { i ->
                    val t0 = SystemClock.elapsedRealtimeNanos()
                    val r = engine.pipeline.inspectCrop(crop, false)
                    val total = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000
                    val p = r.preprocessMs ?: 0; val v = r.visionMs ?: 0
                    pre += p; inf += v; post += (total - p - v).coerceAtLeast(0)
                    w.println("$i,$p,$v,${total - p - v},$total")
                }
            }
            crop.release()
            withContext(Dispatchers.Main) { say("bank $bank: inference ${summary(inf)}; pre ${summary(pre)}; post ${summary(post)}; PSS %.0f MB".format(pssMb())) }
        }
    }
    withContext(engine.visionDispatcher) { engine.loadVision(original) }
}

private suspend fun golden(engine: AppEngine, say: (String) -> Unit) {
    val dir = File(engine.replayDir, "golden")
    val files = replayImages(dir)
    if (files.isEmpty()) { say("No images in ${dir.path}"); return }
    val out = File(engine.resultsDir, "golden_device_bank${engine.bankSize}_${stamp()}.csv")
    withContext(engine.visionDispatcher) {
        out.printWriter().use { w ->
            w.println("file,score_device")
            for (f in files) {
                val rgba = Mat(); Utils.bitmapToMat(decode(f), rgba)
                val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
                val crop = ItemFinder.wholeImage256(rgb)
                val r = engine.pipeline.inspectCrop(crop, false)
                w.println("${f.relativeTo(dir).path.replace('\\', '/')},${r.score}")
                rgba.release(); rgb.release(); crop.release()
            }
        }
    }
    say("Golden scores for ${files.size} images → ${out.name}")
}

private suspend fun fps(engine: AppEngine, context: Context, say: (String) -> Unit) {
    val files = replayImages(engine.replayDir).filter { !it.path.contains("golden") }.take(20)
    if (files.isEmpty()) { say("Push replay images first"); return }
    val bitmaps = withContext(Dispatchers.IO) { files.map { decode(it) to replayReference(engine, it) } }
    var frames = 0; var items = 0
    val t0 = SystemClock.elapsedRealtime()
    withContext(engine.visionDispatcher) {
        while (SystemClock.elapsedRealtime() - t0 < 60_000) {
            val (b, ref) = bitmaps[frames % bitmaps.size]
            val r = engine.pipeline.inspect(b, "fps$frames", ref, keepImages = false)
            frames++; items += r.items.count { it.score != null }
        }
    }
    val secs = (SystemClock.elapsedRealtime() - t0) / 1000.0
    val line = "FPS (vision only, full pipeline incl. finder): %.2f frames/s, %.1f inspected items/min over %.0f s".format(frames / secs, items * 60 / secs, secs)
    File(engine.resultsDir, "bench_fps_bank${engine.bankSize}_${stamp()}.csv").writeText(deviceHeader(context) + "\nframes,items,seconds\n$frames,$items,$secs\n")
    say(line)
}

private suspend fun llamaBench(engine: AppEngine, context: Context, n: Int, say: (String) -> Unit) {
    val cases = advisoryCases(n)
    val out = File(engine.resultsDir, "advisory_test_${stamp()}.csv")
    val pssIdle = pssMb()
    val ttfts = mutableListOf<Long>(); val totals = mutableListOf<Long>(); var valid = 0
    val tps = mutableListOf<Double>()
    withContext(Dispatchers.IO) {
        val load = engine.llama.ensureLoaded()
        val pssLoaded = pssMb()
        out.printWriter().use { w ->
            w.println(deviceHeader(context) + "; llm=llama3.2-1b-instruct 8da4w xnnpack; et=1.5.1; prompt=${PromptBuilder.PROMPT_VERSION}; seed=$ADVISORY_SEED; load_ms=$load; pss_before_mb=%.0f; pss_loaded_mb=%.0f".format(pssIdle, pssLoaded))
            w.println("case,verdict,defect_type,severity,ttft_ms,tokens,total_ms,decode_tok_s,pss_mb,valid,reason,raw,handoff_json")
            cases.forEachIndexed { i, rec ->
                val g = engine.llama.generate(PromptBuilder.llama3Prompt(rec), LlamaEngine.MAX_NEW_TOKENS, 120_000, stopWhen = { PromptBuilder.isComplete(rec, it) })
                val answer = PromptBuilder.completeAnswer(rec, g.text)
                val v = AdvisoryValidator.validate(answer, rec)
                val ok = v is AdvisoryValidator.Result.Valid
                if (ok) valid++
                ttfts += g.ttftMs; totals += g.totalMs
                val decode = if (g.tokens > 1 && g.totalMs > g.ttftMs) (g.tokens - 1) * 1000.0 / (g.totalMs - g.ttftMs) else 0.0
                tps += decode
                w.println(listOf(i + 1, rec.verdict, Csv.escape(rec.defectType), rec.severity, g.ttftMs, g.tokens, g.totalMs, "%.2f".format(decode), "%.0f".format(pssMb()), ok,
                    Csv.escape(if (v is AdvisoryValidator.Result.Invalid) v.reason else ""), Csv.escape(answer), Csv.escape(rec.toJson().toString())).joinToString(","))
                withContext(Dispatchers.Main) { say("case ${i + 1}/$n: ${g.tokens} tok, ttft ${g.ttftMs} ms, total ${g.totalMs} ms, valid=$ok") }
            }
        }
    }
    val med = tps.sorted()[tps.size / 2]
    // Timeout = measured 80-token time (median), doubled (addendum 2.3).
    val per80 = (totals.sorted()[totals.size / 2]).coerceAtLeast(1000)
    engine.timeoutMs = per80 * 2
    say("Llama: TTFT ${summary(ttfts)}; total ${summary(totals)}; decode median %.1f tok/s; valid $valid/$n; timeout set to ${engine.timeoutMs} ms → ${out.name}".format(med))
}

private suspend fun longRun(engine: AppEngine, context: Context, minutes: Int, say: (String) -> Unit) {
    val files = replayImages(engine.replayDir).filter { !it.path.contains("golden") }
    if (files.isEmpty()) { say("Push replay images first"); return }
    val out = File(engine.resultsDir, "longrun_${minutes}min_${stamp()}.csv")
    val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    val t0 = SystemClock.elapsedRealtime()
    var frames = 0
    var nextSample = 0L
    out.printWriter().use { w ->
        w.println(deviceHeader(context))
        w.println("elapsed_s,frames,battery_pct,battery_temp_c,thermal_status,pss_mb,queue_depth")
        while (SystemClock.elapsedRealtime() - t0 < minutes * 60_000L) {
            val f = files[frames % files.size]
            val bmp = withContext(Dispatchers.IO) { decode(f) }
            engine.inspectAndLog(bmp, "longrun:" + f.name, replayReference(engine, f), thumbs = false)
            frames++
            val el = SystemClock.elapsedRealtime() - t0
            if (el >= nextSample) {
                nextSample += 30_000
                val bm = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                val status = pm.currentThermalStatus
                w.println("${el / 1000},$frames,${bm?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)},${(bm?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0},$status,%.0f,${engine.queueDepth()}".format(pssMb()))
                w.flush()
                say("long run ${el / 1000}s frames=$frames thermal=$status${if (status >= PowerManager.THERMAL_STATUS_MODERATE) " WARNING: thermal moderate or worse" else ""}")
            }
        }
    }
    say("Long run done → ${out.name}")
}
