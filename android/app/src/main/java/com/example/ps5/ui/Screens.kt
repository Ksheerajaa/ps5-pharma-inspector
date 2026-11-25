package com.example.ps5.ui

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.example.ps5.AppEngine
import com.example.ps5.LoggedFrame
import com.example.ps5.data.Csv
import com.example.ps5.vision.ItemFinder
import com.example.ps5.vision.ReferenceItem
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

fun stamp(): String = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))

fun replayImages(dir: File): List<File> = dir.walkTopDown()
    .filter { it.isFile && it.extension.lowercase() in setOf("png", "jpg", "jpeg") }
    .sortedBy { it.path }.toList()

/** Folder reference.json {"area_fraction":..,"aspect":..} or "none"; default is the MVTec single-pill reference. */
fun replayReference(engine: AppEngine, file: File): ReferenceItem? {
    var d: File? = file.parentFile
    while (d != null && d.path.startsWith(engine.replayDir.path)) {
        val f = File(d, "reference.json")
        if (f.isFile) {
            val o = JSONObject(f.readText())
            if (o.optBoolean("none", false)) return null
            val rgb = o.optJSONArray("mean_rgb")?.let { a -> DoubleArray(3) { a.getDouble(it) } }
            return ReferenceItem("replay_" + d.name, o.getDouble("area_fraction"), o.getDouble("aspect"), rgb)
        }
        d = d.parentFile
    }
    return engine.defaultReference()
}

fun decode(file: File): Bitmap = BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 })

@Composable
fun ReplayScreen(engine: AppEngine, modifier: Modifier, demoFile: String? = null) {
    val scope = rememberCoroutineScope()
    val files = remember { mutableStateOf(replayImages(engine.replayDir)) }
    var index by remember { mutableIntStateOf(0) }
    var frame by remember { mutableStateOf<LoggedFrame?>(null) }
    var image by remember { mutableStateOf<Bitmap?>(null) }
    var status by remember { mutableStateOf("") }
    var running by remember { mutableStateOf(false) }
    val scroll = rememberScrollState()
    LaunchedEffect(demoFile) {
        val name = demoFile ?: return@LaunchedEffect
        val f = File(engine.replayDir, name)
        if (!f.isFile) { status = "Demo image missing: $name"; return@LaunchedEffect }
        scroll.animateScrollTo(0)
        val bmp = withContext(Dispatchers.IO) { decode(f) }
        frame = engine.inspectAndLog(bmp, "replay:$name", replayReference(engine, f))
        image = bmp; status = name
        delay(3_500)
        scroll.animateScrollTo(scroll.maxValue, androidx.compose.animation.core.tween(2_500))
    }

    Column(modifier.verticalScroll(scroll), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Replay folder: ${engine.replayDir.path}", style = MaterialTheme.typography.bodySmall)
        Text("${files.value.size} images. Same pipeline as the camera.")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { files.value = replayImages(engine.replayDir); index = 0 }) { Text("Rescan") }
            Button(enabled = files.value.isNotEmpty() && !running, onClick = {
                val f = files.value[index % files.value.size]; index++
                scope.launch {
                    val bmp = withContext(Dispatchers.IO) { decode(f) }
                    frame = engine.inspectAndLog(bmp, "replay:" + f.relativeTo(engine.replayDir).path, replayReference(engine, f))
                    image = bmp
                    status = f.relativeTo(engine.replayDir).path
                }
            }) { Text("Next image") }
            Button(enabled = files.value.isNotEmpty() && !running, onClick = {
                running = true
                scope.launch {
                    val out = File(engine.resultsDir, "replay_${stamp()}_bank${engine.bankSize}.csv")
                    out.printWriter().use { w ->
                        w.println("file,frame_status,frame_reason,n_items,item_idx,item_status,item_reason,score,verdict,severity,defect_type,area_pct,location,confidence,frame_ms,preprocess_ms,vision_ms")
                        files.value.forEachIndexed { i, f ->
                            status = "Running ${i + 1}/${files.value.size}"
                            val bmp = withContext(Dispatchers.IO) { decode(f) }
                            val logged = engine.inspectAndLog(bmp, "replay:" + f.relativeTo(engine.replayDir).path, replayReference(engine, f), thumbs = false)
                            val r = logged.result
                            val name = f.relativeTo(engine.replayDir).path.replace('\\', '/')
                            if (r.items.isEmpty()) w.println("$name,${r.status},${r.reason ?: ""},0,,,,,,,,,,,${r.frameMs},,")
                            r.items.forEach { it2 ->
                                val d = it2.decision
                                w.println(listOf(name, r.status, r.reason ?: "", r.items.size, it2.idx, it2.blob.status, Csv.escape(it2.blob.reason ?: ""),
                                    it2.score ?: "", d?.verdict?.label ?: "", d?.severity ?: "", it2.defectType ?: "", it2.features?.areaFraction?.times(100) ?: "",
                                    it2.features?.location ?: "", d?.confidence ?: "", r.frameMs, it2.preprocessMs ?: "", it2.visionMs ?: "").joinToString(","))
                            }
                        }
                    }
                    status = "Wrote ${out.path}"
                    running = false
                }
            }) { Text("Run all → CSV") }
        }
        Text(status)
        frame?.let { FrameResultView(it, image, engine, Modifier.fillMaxWidth()) }
    }
}

@Composable
@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
fun HistoryScreen(engine: AppEngine, modifier: Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    // remember the Flow: creating it during composition restarted the query on every recomposition, so the
    // list never received its first emission while the screen was busy (empty History in the demo recording).
    val rowsFlow = remember { engine.db.dao().recent(300) }
    val rows by rowsFlow.collectAsState(initial = emptyList())
    var filter by remember { mutableStateOf<String?>(null) }
    var msg by remember { mutableStateOf("") }
    Column(modifier, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("Inspection log", style = MaterialTheme.typography.headlineMedium)
        Text("${rows.size} most recent items · stored on this phone", style = MaterialTheme.typography.labelSmall)
        androidx.compose.foundation.layout.FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(null, "PASS", "RE-INSPECT", "REJECT", "attention").forEach { f ->
                if (filter == f) Button(onClick = { filter = f }) { Text(f ?: "All") }
                else OutlinedButton(onClick = { filter = f }) { Text(f ?: "All") }
            }
        }
        if (rows.isEmpty()) Text("No inspections yet. Capture or replay a tray.", style = MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                scope.launch {
                    val out = File(engine.resultsDir, "inspections_${stamp()}.csv")
                    withContext(Dispatchers.IO) { out.writeText(Csv.export(engine.db.dao().all())) }
                    msg = "Exported ${out.path}"
                    val uri = FileProvider.getUriForFile(context, context.packageName + ".files", out)
                    context.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                        type = "text/csv"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }, "Export inspection log"))
                }
            }) { Text("Export CSV") }
        }
        Text(msg, style = MaterialTheme.typography.bodySmall)
        val shown = rows.filter {
            when (filter) {
                null -> true
                "attention" -> it.status != "OK"
                else -> it.verdict == filter
            }
        }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            items(shown, key = { it.id }) { r ->
                Column(Modifier.fillMaxWidth()) {
                    Text("#${r.id} ${r.tsUtc.take(19)} ${r.source.take(40)}", style = MaterialTheme.typography.labelSmall)
                    Text(
                        if (r.status == "OK") "item ${r.itemIdx}: ${r.verdict} · ${r.defectType} · ${r.severity ?: "-"} · score %.3f".format(r.score ?: 0f)
                        else "${r.status}${r.itemIdx?.let { " item $it" } ?: ""}: ${r.statusReason ?: ""}",
                    )
                    r.logText?.let { Text("LOG [${r.advisorySource}${r.advisoryNote?.let { n -> ", $n" } ?: ""}]: $it", style = MaterialTheme.typography.bodySmall) }
                    HorizontalDivider()
                }
            }
        }
    }
}

@Composable
fun AboutScreen(engine: AppEngine, modifier: Modifier) {
    Column(modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("PS5 Pharma Inspector", style = MaterialTheme.typography.titleLarge)
        Text("Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.")
        Text("Vision: PatchCore-lite (ResNet-18 ImageNet features + coreset memory bank), ONNX Runtime 1.23.2, CPU.")
        Text("Advisory: Llama 3.2 1B Instruct, ExecuTorch 1.5.1 (XNNPACK). Verdicts are decided by calibrated rules; Llama only explains and advises.")
        Text("Dataset: MVTec AD (pill), CC BY-NC-SA 4.0, non-commercial use. Defect types are indicative (rule-based).")
        Text("Advice is illustrative engineering guidance, not GMP or regulatory advice.")
        Text("No network permission: the app is fully offline.")
        HorizontalDivider()
        Text("Model versions: ${engine.modelVersions()}", style = MaterialTheme.typography.bodySmall)
        Text("Thresholds: t_pass=%.3f t_reject=%.3f t_pix=%.3f severity cuts=%.4f/%.4f".format(
            engine.config.thresholds.passScore, engine.config.thresholds.rejectScore, engine.config.tPix,
            engine.config.thresholds.lowAreaCut, engine.config.thresholds.highAreaCut,
        ), style = MaterialTheme.typography.bodySmall)
        Text("Llama files expected in: ${engine.modelDir.path}", style = MaterialTheme.typography.bodySmall)
        Text("Reference: ${engine.reference?.id ?: "none"}; default replay reference ${ItemFinder.mvtecDefault(engine.config.targetItemAreaFraction.toDouble()).id}", style = MaterialTheme.typography.bodySmall)
    }
}
