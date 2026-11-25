package com.example.ps5.ui

import android.graphics.Bitmap
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.ps5.AppEngine
import com.example.ps5.LoggedFrame
import com.example.ps5.pipeline.FrameStatus
import com.example.ps5.pipeline.ItemResult
import com.example.ps5.vision.BlobStatus

val PassGreen = Color(0xFF2E9E5B)
val ReinspectAmber = Color(0xFFF2A516)
val RejectRed = Color(0xFFFF4B1F)
val NotInspectedGrey = Color(0xFF8A8A8A)

fun colourFor(item: ItemResult): Color = when {
    item.blob.status != BlobStatus.OK -> NotInspectedGrey
    item.decision?.verdict?.label == "PASS" -> PassGreen
    item.decision?.verdict?.label == "RE-INSPECT" -> ReinspectAmber
    item.decision != null -> RejectRed
    else -> NotInspectedGrey
}

/** Short label shown on the image and on the item chips. */
fun shortLabel(item: ItemResult): String = when (item.blob.status) {
    BlobStatus.OK -> item.decision?.verdict?.label ?: "?"
    BlobStatus.CLUSTER -> "TOUCHING"
    BlobStatus.PARTIAL -> "CUT OFF"
    BlobStatus.UNKNOWN_OBJECT -> "UNKNOWN"
    BlobStatus.TOO_MANY -> "NOT CHECKED"
}

/** Plain-language instruction for items that were not inspected. */
fun attentionHint(status: BlobStatus): String = when (status) {
    BlobStatus.CLUSTER -> "Items are touching. Separate them and capture again."
    BlobStatus.PARTIAL -> "Item is cut by the image edge. Move it to the middle of the tray."
    BlobStatus.UNKNOWN_OBJECT -> "Not the reference product. Remove it or check it by hand."
    BlobStatus.TOO_MANY -> "Tray holds more than 12 items. Remove some and capture again."
    BlobStatus.OK -> ""
}

private fun badImageHint(reason: String?): String = when (reason) {
    "dark" -> "Too dark. Add light over the tray."
    "bright" -> "Too bright. Use the matte dark tray and reduce direct light."
    "glare" -> "Glare spot. Move the light or tilt it so there is no reflection."
    "blur" -> "Blurred. Hold the phone still or wait for the scene to settle."
    else -> "Image not usable. Capture again."
}

private data class Banner(val title: String, val subtitle: String, val colour: Color)

private fun banner(frame: LoggedFrame): Banner {
    val r = frame.result
    return when (r.status) {
        FrameStatus.NO_ITEMS -> Banner("NO ITEMS", "Place tablets on the tray.", NotInspectedGrey)
        FrameStatus.BAD_IMAGE -> Banner("RETAKE PHOTO", badImageHint(r.reason), NotInspectedGrey)
        FrameStatus.OK -> {
            val rejects = r.items.count { it.decision?.verdict?.label == "REJECT" }
            val reinspect = r.items.count { it.decision?.verdict?.label == "RE-INSPECT" }
            val attention = r.items.count { it.blob.status != BlobStatus.OK }
            when {
                rejects > 0 -> Banner("REJECT: $rejects of ${r.items.size}", "Remove the red items from the tray.", RejectRed)
                reinspect > 0 || attention > 0 -> Banner("CHECK ${reinspect + attention} ITEM${if (reinspect + attention > 1) "S" else ""}", "Look at the amber and grey items before release.", ReinspectAmber)
                else -> Banner("ALL PASS", "${r.items.size} item${if (r.items.size > 1) "s" else ""} inspected. Tray can be released.", PassGreen)
            }
        }
    }
}

/** Kept for callers that log or share a one-line summary. */
fun batchBanner(frame: LoggedFrame): String = banner(frame).let { "${it.title}: ${it.subtitle}" }

@Composable
private fun BigStat(n: Int, label: String, colour: Color) {
    Column {
        Text("$n", color = colour, fontSize = 38.sp, fontWeight = FontWeight.Light, letterSpacing = (-1).sp)
        Text(label, color = colour.copy(alpha = 0.7f), style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun CountChip(label: String, n: Int, colour: Color) {
    Text(
        "$n $label", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp,
        modifier = Modifier.clip(RoundedCornerShape(50)).background(colour.copy(alpha = if (n > 0) 1f else 0.35f))
            .padding(horizontal = 10.dp, vertical = 3.dp),
    )
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun FrameResultView(frame: LoggedFrame, image: Bitmap?, engine: AppEngine, modifier: Modifier = Modifier) {
    val advisory by engine.advisory.collectAsState()
    val r = frame.result
    var selected by remember(frame) {
        mutableIntStateOf(
            r.items.indexOfFirst { it.decision?.verdict?.label == "REJECT" }.takeIf { it >= 0 }
                ?: r.items.indexOfFirst { it.decision != null && it.decision.verdict.label != "PASS" || it.blob.status != BlobStatus.OK }.coerceAtLeast(0),
        )
    }
    val b = banner(frame)
    val haptics = LocalHapticFeedback.current
    LaunchedEffect(frame) {
        if (r.items.any { it.decision?.verdict?.label == "REJECT" }) haptics.performHapticFeedback(HapticFeedbackType.LongPress)
    }

    Column(modifier, verticalArrangement = Arrangement.spacedBy(10.dp)) {
        // Headline banner: one decision the operator can act on.
        val onBanner = if (b.colour == PassGreen || b.colour == NotInspectedGrey) Ink else Color.White
        val bg = when (b.colour) { PassGreen -> CardWhite; NotInspectedGrey -> CardWhite; else -> b.colour }
        Column(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(28.dp)).background(bg).padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Box(Modifier.size(8.dp).clip(RoundedCornerShape(50)).background(if (bg == CardWhite) b.colour else Color.White))
                Text("TRAY RESULT · ${r.frameMs} MS · OFFLINE", color = onBanner.copy(alpha = 0.7f), style = MaterialTheme.typography.labelSmall)
            }
            Text(b.title.lowercase().replaceFirstChar { it.uppercase() }, color = onBanner, style = MaterialTheme.typography.headlineLarge)
            Text(b.subtitle, color = onBanner.copy(alpha = 0.85f), fontSize = 14.sp)
            if (r.status == FrameStatus.OK) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    BigStat(r.items.count { it.decision?.verdict?.label == "PASS" }, "PASS", onBanner)
                    BigStat(r.items.count { it.decision?.verdict?.label == "RE-INSPECT" }, "CHECK", onBanner)
                    BigStat(r.items.count { it.decision?.verdict?.label == "REJECT" }, "REJECT", onBanner)
                    BigStat(r.items.count { it.blob.status != BlobStatus.OK }, "SKIPPED", onBanner)
                }
            }
        }

        // Image with a labelled box per item.
        if (image != null) {
            Box(Modifier.fillMaxWidth().aspectRatio(image.width.toFloat() / image.height).clip(RoundedCornerShape(8.dp))) {
                Image(image.asImageBitmap(), null, Modifier.fillMaxSize())
                Canvas(Modifier.fillMaxSize()) {
                    val sx = size.width / r.width; val sy = size.height / r.height
                    val text = android.graphics.Paint().apply { color = android.graphics.Color.WHITE; textSize = 30f; isFakeBoldText = true; isAntiAlias = true }
                    r.items.forEachIndexed { i, item ->
                        val box = item.blob.box
                        val c = colourFor(item)
                        drawRect(c, Offset(box.x * sx, box.y * sy), Size(box.width * sx, box.height * sy), style = Stroke(width = if (i == selected) 9f else 5f))
                        val label = "${item.idx} ${shortLabel(item)}"
                        val w = text.measureText(label) + 16f
                        val x = box.x * sx; val y = (box.y * sy - 40f).coerceAtLeast(0f)
                        drawRect(c, Offset(x, y), Size(w, 40f))
                        drawContext.canvas.nativeCanvas.drawText(label, x + 8f, y + 30f, text)
                    }
                }
            }
        }

        // Item selector.
        if (r.items.isNotEmpty()) {
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                r.items.forEachIndexed { i, item ->
                    Text(
                        "#${item.idx} ${shortLabel(item)}", color = Color.White, fontWeight = FontWeight.SemiBold, fontSize = 13.sp,
                        modifier = Modifier.clip(RoundedCornerShape(50)).background(colourFor(item))
                            .then(if (i == selected) Modifier.border(3.dp, Ink, RoundedCornerShape(50)) else Modifier)
                            .clickable { selected = i }.padding(horizontal = 10.dp, vertical = 6.dp),
                    )
                }
            }
            r.items.getOrNull(selected)?.let { ItemCard(it, frame.rowIds.getOrNull(selected)?.let { id -> advisory[id] }) }
        }
    }
}

@Composable
private fun Fact(label: String, value: String) {
    Row {
        Text(label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.width(84.dp))
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
    }
}

@Composable
fun ItemCard(item: ItemResult, adv: com.example.ps5.AdvisoryState?) {
    val c = colourFor(item)
    Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(24.dp), colors = CardDefaults.cardColors(containerColor = CardWhite)) {
        // Verdict strip.
        Row(Modifier.fillMaxWidth().background(c).padding(horizontal = 12.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Item ${item.idx}", color = Color.White, fontSize = 16.sp, modifier = Modifier.weight(1f))
            Text(
                if (item.blob.status == BlobStatus.OK) item.decision?.verdict?.label ?: "" else "NOT INSPECTED",
                color = Color.White, fontWeight = FontWeight.ExtraBold, fontSize = 20.sp,
            )
        }
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (item.blob.status != BlobStatus.OK) {
                Text(attentionHint(item.blob.status), fontWeight = FontWeight.Bold)
                item.blob.reason?.let { Text("Reason: $it", style = MaterialTheme.typography.bodySmall) }
                Text("This item was never marked PASS.", style = MaterialTheme.typography.bodySmall)
                return@Column
            }
            val d = item.decision ?: return@Column
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                item.crop?.let { crop ->
                    Box(Modifier.size(128.dp).clip(RoundedCornerShape(8.dp))) {
                        Image(crop.asImageBitmap(), null, Modifier.fillMaxSize())
                        item.heat?.let { Image(it.asImageBitmap(), "defect heatmap", Modifier.fillMaxSize()) }
                    }
                }
                Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    Fact("Defect", if (d.verdict.label == "PASS") "none" else "${item.defectType}${if (item.typeBasis != null) " *" else ""}")
                    Fact("Severity", d.severity?.name ?: "-")
                    Fact("Where", item.features?.location ?: "-")
                    Fact("Confidence", "${(d.confidence * 100).toInt()}%")
                    Fact("Score", "%.2f".format(item.score))
                    Fact("Vision", "${item.visionMs} ms")
                }
            }
            if (item.typeBasis != null) Text("* defect type is indicative (rule-based)", style = MaterialTheme.typography.labelSmall)
            if (d.verdict.label != "PASS") AdviceCard(adv)
        }
    }
}

@Composable
private fun AdviceCard(adv: com.example.ps5.AdvisoryState?) = androidx.compose.runtime.CompositionLocalProvider(
    androidx.compose.material3.LocalContentColor provides Color.White,
) {
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(Ink).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        when {
            adv == null || (adv.pending && adv.streaming.isNullOrBlank()) -> {
                Text("Operator advice", fontWeight = FontWeight.Bold)
                Text("Llama is preparing advice on the phone…", style = MaterialTheme.typography.bodySmall)
                LinearProgressIndicator(Modifier.fillMaxWidth())
            }
            adv.pending -> {
                Text("Operator advice · writing…", fontWeight = FontWeight.Bold)
                Text(adv.streaming ?: "", style = MaterialTheme.typography.bodyMedium)
            }
            adv.text != null -> {
                val ai = adv.source == "llm"
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Operator advice", fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                    Text(
                        if (ai) "Llama · on-device" else "Standard text",
                        color = Color.White, fontSize = 11.sp,
                        modifier = Modifier.clip(RoundedCornerShape(50)).background(if (ai) Signal else NotInspectedGrey).padding(horizontal = 8.dp, vertical = 2.dp),
                    )
                }
                listOf("Do" to adv.text.action, "Why" to adv.text.reason, "Check" to adv.text.check, "Log" to adv.text.log).forEach { (k, v) ->
                    Row {
                        Text(k, fontWeight = FontWeight.Bold, modifier = Modifier.width(52.dp))
                        Text(v, style = MaterialTheme.typography.bodyMedium)
                    }
                }
                if (!ai && adv.note != null) Text("Llama text not used: ${adv.note}", style = MaterialTheme.typography.labelSmall)
            }
        }
    }
}
