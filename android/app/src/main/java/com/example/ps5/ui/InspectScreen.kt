package com.example.ps5.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.ps5.AppEngine
import com.example.ps5.LoggedFrame
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/** Stop-and-shoot trigger (addendum 2): scene stable ~0.5 s and changed since the last inspection. */
class StabilityTrigger {
    private var prev: IntArray? = null
    private var lastInspected: IntArray? = null
    private var stableSince = -1L
    var lastMotion = 0.0; private set

    fun signature(bmp: Bitmap): IntArray {
        val s = Bitmap.createScaledBitmap(bmp, 16, 16, true)
        val px = IntArray(256); s.getPixels(px, 0, 16, 0, 0, 16, 16)
        return IntArray(256) { (((px[it] shr 16) and 255) * 299 + ((px[it] shr 8) and 255) * 587 + (px[it] and 255) * 114) / 1000 }
    }

    private fun diff(a: IntArray, b: IntArray) = a.indices.sumOf { abs(a[it] - b[it]) }.toDouble() / a.size

    /** Returns true when an automatic inspection should fire for this frame. */
    fun update(sig: IntArray, now: Long): Boolean {
        val p = prev
        prev = sig
        if (p == null) return false
        lastMotion = diff(sig, p)
        if (lastMotion > STABLE_DIFF) { stableSince = -1; return false }
        if (stableSince < 0) stableSince = now
        if (now - stableSince < STABLE_MS) return false
        val li = lastInspected
        return li == null || diff(sig, li) > CHANGE_DIFF
    }

    fun markInspected(sig: IntArray) { lastInspected = sig }

    companion object {
        const val STABLE_DIFF = 3.0   // mean abs luma difference between consecutive 16x16 signatures
        const val CHANGE_DIFF = 8.0   // difference from the last inspected scene that counts as a new tray
        const val STABLE_MS = 500L
    }
}

@Composable
fun InspectScreen(engine: AppEngine, modifier: Modifier) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    val scope = rememberCoroutineScope()
    var frame by remember { mutableStateOf<LoggedFrame?>(null) }
    var image by remember { mutableStateOf<Bitmap?>(null) }
    var message by remember { mutableStateOf<String?>(null) }
    var auto by remember { mutableStateOf(false) }
    var settingReference by remember { mutableStateOf(false) }
    val captureRequested = remember { AtomicBoolean(false) }
    val busy = remember { AtomicBoolean(false) }
    val trigger = remember { StabilityTrigger() }
    val autoFlag = remember { AtomicBoolean(false) }
    autoFlag.set(auto)

    if (!engine.visionReady()) {
        Text(engine.visionError ?: "Vision model unavailable. No inspection is possible.", color = MaterialTheme.colorScheme.error, modifier = modifier)
        return
    }
    if (!granted) {
        Column(modifier, verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Camera permission is required for live inspection.", style = MaterialTheme.typography.titleMedium)
            Text("Tap Allow camera. If you denied it permanently: Settings > Apps > PS5 Pharma Inspector > Permissions > Camera > Allow. Replay mode still works without the camera.")
            Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) { Text("Allow camera") }
        }
        return
    }

    fun handle(bmp: Bitmap, sig: IntArray) {
        if (!busy.compareAndSet(false, true)) return
        trigger.markInspected(sig)
        val asReference = settingReference
        scope.launch {
            try {
                val ref = if (asReference) null else engine.reference
                val logged = engine.inspectAndLog(bmp, "camera", ref)
                image = bmp
                frame = logged
                message = if (asReference) {
                    settingReference = false
                    engine.setReferenceFrom(logged.result) ?: "Reference item stored (${engine.reference?.id})."
                } else if (engine.reference == null) "No reference item set: size checks use no reference. Use Set reference item." else null
            } finally {
                busy.set(false)
            }
        }
    }

    Column(modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            "Offline · ${engine.productLabel()} · bank ${engine.bankSize} · ref: ${engine.reference?.id ?: "none"} · " +
                if (engine.llama.filesPresent()) "Llama ready" else "advisory model unavailable (template advice)",
            style = MaterialTheme.typography.bodySmall,
        )
        CameraAnalysis(Modifier.fillMaxWidth().height(300.dp)) { bmp ->
            val sig = trigger.signature(bmp)
            val fire = trigger.update(sig, SystemClock.elapsedRealtime())
            if (captureRequested.getAndSet(false) || (autoFlag.get() && fire)) handle(bmp, sig)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            Button(onClick = { captureRequested.set(true) }) { Text("Capture") }
            Text("Auto")
            Switch(checked = auto, onCheckedChange = { auto = it })
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { settingReference = true; captureRequested.set(true) }) { Text("Set reference item") }
            OutlinedButton(onClick = { engine.clearReference(); message = "Reference cleared." }) { Text("Clear ref") }
        }
        message?.let { Text(it, color = MaterialTheme.colorScheme.primary) }
        frame?.let { FrameResultView(it, image, engine, Modifier.fillMaxWidth()) }
    }
}

@Composable
private fun CameraAnalysis(modifier: Modifier, onFrame: (Bitmap) -> Unit) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    var error by remember { mutableStateOf<String?>(null) }
    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FIT_CENTER } }
    DisposableEffect(owner) {
        val executor = Executors.newSingleThreadExecutor()
        var provider: ProcessCameraProvider? = null
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try {
                val p = future.get(); provider = p
                val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder().setResolutionStrategy(
                            ResolutionStrategy(Size(1280, 960), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
                        ).build(),
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(executor) { proxy ->
                    try {
                        val raw = proxy.toBitmap()
                        val rot = proxy.imageInfo.rotationDegrees
                        val bmp = if (rot == 0) raw else Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(rot.toFloat()) }, true)
                        onFrame(bmp)
                    } finally {
                        proxy.close()
                    }
                }
                p.unbindAll()
                p.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (e: Exception) {
                error = e.message ?: e.javaClass.simpleName
            }
        }, ContextCompat.getMainExecutor(context))
        onDispose {
            provider?.unbindAll()
            executor.shutdown()
        }
    }
    Column(modifier) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxWidth().weight(1f))
        error?.let { Text("Camera unavailable: $it") }
    }
}
