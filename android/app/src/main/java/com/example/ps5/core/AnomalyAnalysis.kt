package com.example.ps5.core

import kotlin.math.sqrt

data class AnomalyFeatures(
    val peak: Float,
    /** Share of the crop with anomaly >= t_pix. */
    val areaFraction: Float,
    val blobCount: Int,
    /** Largest blob box in crop pixels: x, y, w, h. Null when no pixel exceeds t_pix. */
    val box: IntArray?,
    val centroidX: Float,
    val centroidY: Float,
    val location: String,
    /** sqrt(lambda_max / lambda_min) of the largest blob's pixel covariance. */
    val elongation: Float,
    /** Mask of the largest blob, row-major, size*size. */
    val largestMask: BooleanArray?,
)

enum class DefectType(val label: String) {
    CRACK_OR_SCRATCH("crack or scratch"),
    CONTAMINATION_OR_COLOUR("contamination or colour fault"),
    DEFORMATION("deformation"),
    UNSPECIFIED("unspecified anomaly"),
}

object AnomalyAnalysis {
    fun locationWord(cx: Float, cy: Float, size: Int): String {
        val col = (cx / size * 3).toInt().coerceIn(0, 2)
        val row = (cy / size * 3).toInt().coerceIn(0, 2)
        return arrayOf(
            arrayOf("upper-left", "top", "upper-right"),
            arrayOf("left", "centre", "right"),
            arrayOf("lower-left", "bottom", "lower-right"),
        )[row][col]
    }

    fun analyse(map: FloatArray, size: Int, tPix: Float): AnomalyFeatures {
        require(map.size == size * size)
        val above = BooleanArray(map.size) { map[it] >= tPix }
        var peak = Float.NEGATIVE_INFINITY
        var count = 0
        for (i in map.indices) {
            if (map[i] > peak) peak = map[i]
            if (above[i]) count++
        }
        val labels = IntArray(map.size)
        var blobs = 0
        var bestLabel = 0
        var bestSize = 0
        val stack = IntArray(map.size)
        for (start in map.indices) {
            if (!above[start] || labels[start] != 0) continue
            blobs++
            var sp = 0
            var n = 0
            stack[sp++] = start
            labels[start] = blobs
            while (sp > 0) {
                val p = stack[--sp]
                n++
                val x = p % size
                val y = p / size
                if (x > 0) push(p - 1, above, labels, blobs, stack, sp).also { sp = it }
                if (x < size - 1) push(p + 1, above, labels, blobs, stack, sp).also { sp = it }
                if (y > 0) push(p - size, above, labels, blobs, stack, sp).also { sp = it }
                if (y < size - 1) push(p + size, above, labels, blobs, stack, sp).also { sp = it }
            }
            if (n > bestSize) {
                bestSize = n
                bestLabel = blobs
            }
        }
        if (bestLabel == 0) {
            return AnomalyFeatures(peak, 0f, 0, null, size / 2f, size / 2f, "none", 1f, null)
        }
        var minX = size; var minY = size; var maxX = 0; var maxY = 0
        var sx = 0.0; var sy = 0.0
        val mask = BooleanArray(map.size)
        for (i in map.indices) {
            if (labels[i] != bestLabel) continue
            mask[i] = true
            val x = i % size; val y = i / size
            if (x < minX) minX = x; if (x > maxX) maxX = x
            if (y < minY) minY = y; if (y > maxY) maxY = y
            sx += x; sy += y
        }
        val cx = sx / bestSize; val cy = sy / bestSize
        var cxx = 0.0; var cyy = 0.0; var cxy = 0.0
        for (i in map.indices) {
            if (!mask[i]) continue
            val dx = i % size - cx; val dy = i / size - cy
            cxx += dx * dx; cyy += dy * dy; cxy += dx * dy
        }
        cxx /= bestSize; cyy /= bestSize; cxy /= bestSize
        val tr = cxx + cyy
        val disc = sqrt(((cxx - cyy) * (cxx - cyy)) / 4 + cxy * cxy)
        val l1 = tr / 2 + disc
        val l2 = (tr / 2 - disc).coerceAtLeast(1e-6)
        return AnomalyFeatures(
            peak = peak,
            areaFraction = count.toFloat() / map.size,
            blobCount = blobs,
            box = intArrayOf(minX, minY, maxX - minX + 1, maxY - minY + 1),
            centroidX = cx.toFloat(),
            centroidY = cy.toFloat(),
            location = locationWord(cx.toFloat(), cy.toFloat(), size),
            elongation = sqrt(l1 / l2).toFloat(),
            largestMask = mask,
        )
    }

    private fun push(q: Int, above: BooleanArray, labels: IntArray, label: Int, stack: IntArray, sp: Int): Int {
        if (!above[q] || labels[q] != 0) return sp
        labels[q] = label
        stack[sp] = q
        return sp + 1
    }
}

/**
 * Tier 1 indicative defect typing (addendum section 4). Constants are fixed, not tuned on eval.
 */
object DefectTyper {
    const val ELONGATION_CRACK = 3.0f
    const val COLOUR_DELTA_E = 12.0
    data class Typed(val type: DefectType, val basis: String)

    fun classify(features: AnomalyFeatures, colourDeltaE: Double, highAreaCut: Float): Typed {
        val basis = "elongation=%.2f, area=%.2f%%, dE=%.1f".format(
            features.elongation, features.areaFraction * 100, colourDeltaE,
        )
        val type = when {
            features.box == null -> DefectType.UNSPECIFIED
            features.elongation >= ELONGATION_CRACK -> DefectType.CRACK_OR_SCRATCH
            colourDeltaE >= COLOUR_DELTA_E -> DefectType.CONTAMINATION_OR_COLOUR
            features.areaFraction >= highAreaCut -> DefectType.DEFORMATION
            else -> DefectType.UNSPECIFIED
        }
        return Typed(type, basis)
    }
}
