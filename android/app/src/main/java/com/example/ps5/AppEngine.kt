package com.example.ps5

import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.example.ps5.advisory.AdvisoryRecord
import com.example.ps5.advisory.AdvisoryTemplates
import com.example.ps5.advisory.AdvisoryText
import com.example.ps5.advisory.AdvisoryValidator
import com.example.ps5.advisory.LlamaEngine
import com.example.ps5.advisory.PromptBuilder
import com.example.ps5.core.CalibratedConfig
import com.example.ps5.core.Verdict
import com.example.ps5.data.AppDb
import com.example.ps5.data.InspectionEntity
import com.example.ps5.pipeline.FrameResult
import com.example.ps5.pipeline.FrameStatus
import com.example.ps5.pipeline.InspectionPipeline
import com.example.ps5.pipeline.ItemResult
import com.example.ps5.vision.BlobStatus
import com.example.ps5.vision.ItemFinder
import com.example.ps5.vision.ReferenceItem
import com.example.ps5.vision.VisionModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import java.io.File
import java.time.Instant
import java.util.PriorityQueue
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

data class AdvisoryState(
    val rowId: Long,
    val itemNo: Int,
    val verdict: String,
    val streaming: String = "",
    val text: AdvisoryText? = null,
    val source: String? = null,
    val note: String? = null,
    val pending: Boolean = true,
)

data class LoggedFrame(val result: FrameResult, val rowIds: List<Long>)

/** App-wide singleton: models, database, advisory worker. Vision and Llama each have one thread. */
class AppEngine private constructor(private val context: Context) {
    val sessionId: String = UUID.randomUUID().toString().take(8)
    val db = AppDb.get(context)
    val visionDispatcher = Executors.newSingleThreadExecutor { Thread(it, "vision") }.asCoroutineDispatcher()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val prefs = context.getSharedPreferences("ps5", Context.MODE_PRIVATE)

    val modelDir: File = File(context.getExternalFilesDir(null), "models").apply { mkdirs() }
    val replayDir: File = File(context.getExternalFilesDir(null), "replay").apply { mkdirs() }
    val resultsDir: File = File(context.getExternalFilesDir(null), "results").apply { mkdirs() }
    val llama = LlamaEngine(modelDir)

    var bankSize: Int = prefs.getInt("bank", DEFAULT_BANK)
        private set
    /** Products taught after the pill model live in <external files>/products/<name>/{model.onnx,config.json}. */
    val productsDir: File = File(context.getExternalFilesDir(null), "products").apply { mkdirs() }
    var product by androidx.compose.runtime.mutableStateOf(prefs.getString("product", "pill") ?: "pill")
        private set

    fun products(): List<String> = listOf("pill") + (productsDir.listFiles()?.filter {
        File(it, "model.onnx").isFile && File(it, "config.json").isFile
    }?.map { it.name }?.sorted() ?: emptyList())

    fun productLabel(): String = product.replaceFirstChar { it.uppercase() }

    fun selectProduct(name: String) {
        product = name
        prefs.edit().putString("product", name).apply()
        loadVision(bankSize)
        clearReference()
    }
    lateinit var config: CalibratedConfig
        private set
    lateinit var pipeline: InspectionPipeline
        private set
    private var model: VisionModel? = null
    var visionError: String? = null
        private set

    private val _advisory = MutableStateFlow<Map<Long, AdvisoryState>>(emptyMap())
    val advisory: StateFlow<Map<Long, AdvisoryState>> = _advisory

    var reference: ReferenceItem? = loadReference()
        private set

    init {
        check(OpenCVLoader.initLocal()) { "OpenCV native library failed to load" }
        loadVision(bankSize)
    }

    fun visionReady() = visionError == null && model != null

    fun loadVision(bank: Int) {
        runCatching {
            model?.close()
            val dir = File(productsDir, product)
            val custom = product != "pill" && File(dir, "model.onnx").isFile
            val json = if (custom) File(dir, "config.json").readText()
                else context.assets.open("config/thresholds_pill_bank$bank.json").bufferedReader().readText()
            config = CalibratedConfig.parse(json)
            val bytes = if (custom) File(dir, "model.onnx").readBytes() else context.assets.open("vision/patchcore_pill_$bank.onnx").readBytes()
            model = VisionModel(bytes)
            pipeline = InspectionPipeline(model!!, config)
            bankSize = bank
            prefs.edit().putInt("bank", bank).apply()
            visionError = null
        }.onFailure {
            model = null
            visionError = "Vision model unavailable: ${it.message}"
            Log.e(TAG, "vision load failed", it)
        }
    }

    fun modelVersions(): String =
        "vision=${if (product == "pill") "patchcore_pill_$bankSize.onnx" else "products/$product/model.onnx"};ort=1.23.2;thresholds=${config.createdUtc};llm=${if (llama.filesPresent()) "llama3.2-1b-instruct.pte" else "missing"};et=1.5.1"

    // ---- reference item -------------------------------------------------------------------------
    fun defaultReference() = ItemFinder.mvtecDefault(config.targetItemAreaFraction.toDouble())

    fun setReferenceFrom(result: FrameResult): String? {
        val ok = result.items.map { it.blob }.filter { it.status != BlobStatus.PARTIAL }
        if (ok.size != 1) return "Place exactly one good item fully inside the frame (found ${ok.size})."
        val b = ok.first()
        reference = ReferenceItem("ref_" + Instant.now().epochSecond, b.areaFraction, b.aspect, b.meanRgb)
        prefs.edit().putString("reference", JSONObject().apply {
            put("id", reference!!.id); put("area", b.areaFraction); put("aspect", b.aspect)
            put("r", b.meanRgb[0]); put("g", b.meanRgb[1]); put("b", b.meanRgb[2])
        }.toString()).apply()
        return null
    }

    fun clearReference() {
        reference = null
        prefs.edit().remove("reference").apply()
    }

    private fun loadReference(): ReferenceItem? = prefs.getString("reference", null)?.let {
        val o = JSONObject(it)
        ReferenceItem(o.getString("id"), o.getDouble("area"), o.getDouble("aspect"), doubleArrayOf(o.getDouble("r"), o.getDouble("g"), o.getDouble("b")))
    }

    // ---- inspection -----------------------------------------------------------------------------
    /** Runs the pipeline on the vision thread, logs rows, queues advisories. */
    suspend fun inspectAndLog(bitmap: Bitmap, source: String, reference: ReferenceItem?, thumbs: Boolean = true): LoggedFrame {
        val frameId = "f" + System.currentTimeMillis().toString(36) + frameCounter.incrementAndGet()
        val result = withContext(visionDispatcher) { pipeline.inspect(bitmap, frameId, reference, keepImages = thumbs) }
        val rows = toRows(result, source, reference)
        val ids = db.dao().insertAll(rows)
        if (result.status == FrameStatus.OK) {
            result.items.zip(ids).forEach { (item, id) -> scheduleAdvisory(item, id) }
        }
        return LoggedFrame(result, ids)
    }

    private fun toRows(r: FrameResult, source: String, reference: ReferenceItem?): List<InspectionEntity> {
        val ts = Instant.now().toString()
        val versions = modelVersions()
        if (r.status != FrameStatus.OK) {
            return listOf(
                InspectionEntity(
                    tsUtc = ts, sessionId = sessionId, source = source, product = product, frameId = r.frameId,
                    itemIdx = null, nItemsInFrame = 0, status = r.status.name, statusReason = r.reason,
                    referenceId = reference?.id, score = null, thresholdPass = config.thresholds.passScore,
                    thresholdReject = config.thresholds.rejectScore, verdict = null, severity = null,
                    defectType = null, typeBasis = null, areaPct = null, bbox = null, defectBox = null,
                    location = null, confidence = null, handoffJson = null, visionMs = null, preprocessMs = null,
                    frameMs = r.frameMs, thumbPath = null, modelVersions = versions,
                ),
            )
        }
        return r.items.map { item ->
            val d = item.decision
            val record = d?.let { record(item) }
            InspectionEntity(
                tsUtc = ts, sessionId = sessionId, source = source, product = product, frameId = r.frameId,
                itemIdx = item.idx, nItemsInFrame = r.items.size, status = item.blob.status.name,
                statusReason = item.blob.reason, referenceId = reference?.id, score = item.score,
                thresholdPass = config.thresholds.passScore, thresholdReject = config.thresholds.rejectScore,
                verdict = d?.verdict?.label, severity = d?.severity?.name, defectType = item.defectType,
                typeBasis = item.typeBasis, areaPct = item.features?.areaFraction?.times(100),
                bbox = item.blob.box.let { "${it.x} ${it.y} ${it.width} ${it.height}" },
                defectBox = item.features?.box?.joinToString(" "), location = item.features?.location,
                confidence = d?.confidence, handoffJson = record?.toJson()?.toString(),
                visionMs = item.visionMs, preprocessMs = item.preprocessMs, frameMs = r.frameMs,
                thumbPath = null, modelVersions = versions,
            )
        }
    }

    private fun record(item: ItemResult, recentRejects: Int = 0): AdvisoryRecord {
        val d = item.decision!!
        return AdvisoryRecord(
            product = product, itemNo = item.idx, verdict = d.verdict.label, defectType = item.defectType ?: "none",
            severity = d.severity?.name ?: "NONE", confidencePct = (d.confidence * 100).toInt(),
            areaPct = (item.features?.areaFraction ?: 0f) * 100.0, location = item.features?.location ?: "none",
            box = item.features?.box, recentRejectsLast20 = recentRejects,
        )
    }

    // ---- advisory queue (addendum 2.3) ------------------------------------------------------------
    private data class Job(val rowId: Long, val record: AdvisoryRecord, val priority: Int, val seq: Long, val enqueuedAt: Long)

    private val queue = PriorityQueue<Job>(compareBy<Job>({ it.priority }, { it.seq }))
    private val queueLock = Object()
    private val seq = AtomicLong()
    private val frameCounter = AtomicLong()
    @Volatile var llmEnabled = true
    var timeoutMs: Long
        get() = prefs.getLong("llm_timeout", DEFAULT_TIMEOUT_MS)
        set(v) = prefs.edit().putLong("llm_timeout", v).apply()

    private val worker = Thread({ workerLoop() }, "llama").apply { isDaemon = true; start() }

    private fun scheduleAdvisory(item: ItemResult, rowId: Long) {
        val d = item.decision ?: return
        scope.launch {
            val rec = record(item, db.dao().rejectsInLast20())
            val state = AdvisoryState(rowId, item.idx, d.verdict.label)
            when {
                d.verdict == Verdict.PASS -> finishTemplate(rowId, rec, "pass_no_llm", null)
                !llmEnabled || !llama.filesPresent() -> finishTemplate(rowId, rec, "model_missing", null)
                else -> synchronized(queueLock) {
                    if (queue.size >= QUEUE_CAPACITY) {
                        scope.launch { finishTemplate(rowId, rec, "queue_full", null) }
                    } else {
                        publish(state)
                        val priority = if (d.verdict == Verdict.REJECT) 0 else 1
                        queue.add(Job(rowId, rec, priority, seq.incrementAndGet(), SystemClock.elapsedRealtime()))
                        queueLock.notifyAll()
                    }
                }
            }
        }
    }

    private fun workerLoop() {
        while (true) {
            val job = synchronized(queueLock) {
                while (queue.isEmpty()) queueLock.wait()
                queue.poll()
            }
            val wait = SystemClock.elapsedRealtime() - job.enqueuedAt
            try {
                val prompt = PromptBuilder.llama3Prompt(job.record)
                val streamed = StringBuilder(PromptBuilder.prefill(job.record))
                val g = llama.generate(prompt, LlamaEngine.MAX_NEW_TOKENS, timeoutMs, stopWhen = { PromptBuilder.isComplete(job.record, it) }) { tok ->
                    streamed.append(tok)
                    update(job.rowId) { it.copy(streaming = streamed.toString()) }
                }
                val answer = PromptBuilder.completeAnswer(job.record, g.text)
                val validation = AdvisoryValidator.validate(answer, job.record)
                val (text, source, note) = when {
                    g.timedOut -> Triple(AdvisoryTemplates.forRecord(job.record), "template", "timeout")
                    g.error != null -> Triple(AdvisoryTemplates.forRecord(job.record), "template", "error:${g.error}")
                    validation is AdvisoryValidator.Result.Valid -> Triple(validation.text, "llm", null)
                    else -> Triple(AdvisoryTemplates.forRecord(job.record), "template", "validator:" + (validation as AdvisoryValidator.Result.Invalid).reason)
                }
                runBlocking {
                    db.dao().updateAdvisory(job.rowId, text.action, text.reason, text.check, text.log, source, note, wait, g.loadMs, g.ttftMs, g.tokens, g.totalMs, answer)
                }
                update(job.rowId) { it.copy(text = text, source = source, note = note, pending = false) }
            } catch (t: Throwable) {
                Log.e(TAG, "advisory failed", t)
                runBlocking { finishTemplate(job.rowId, job.record, "error:${t.message}", wait) }
            }
        }
    }

    private suspend fun finishTemplate(rowId: Long, rec: AdvisoryRecord, note: String, wait: Long?) {
        val t = AdvisoryTemplates.forRecord(rec)
        db.dao().updateAdvisory(rowId, t.action, t.reason, t.check, t.log, "template", note, wait, null, null, null, null, null)
        publish(AdvisoryState(rowId, rec.itemNo, rec.verdict, text = t, source = "template", note = note, pending = false))
    }

    private fun publish(s: AdvisoryState) { _advisory.value = _advisory.value + (s.rowId to s) }
    private fun update(rowId: Long, f: (AdvisoryState) -> AdvisoryState) {
        val cur = _advisory.value[rowId] ?: return
        _advisory.value = _advisory.value + (rowId to f(cur))
    }

    fun queueDepth() = synchronized(queueLock) { queue.size }

    companion object {
        const val TAG = "PS5"
        const val DEFAULT_BANK = 4096
        const val QUEUE_CAPACITY = 5
        const val DEFAULT_TIMEOUT_MS = 30_000L
        val BANKS = listOf(256, 1024, 4096)
        @Volatile private var instance: AppEngine? = null
        fun get(context: Context): AppEngine = instance ?: synchronized(this) {
            instance ?: AppEngine(context.applicationContext).also { instance = it }
        }
    }
}
