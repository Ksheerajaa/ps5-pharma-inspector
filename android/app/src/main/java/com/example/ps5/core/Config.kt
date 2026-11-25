package com.example.ps5.core

import org.json.JSONObject

/** Calibrated constants from ml/scripts/06_calibrate.py (val_good + calib only). */
data class CalibratedConfig(
    val category: String,
    val bankSize: Int,
    val checkpoint: String,
    val createdUtc: String,
    val splitSeed: Int,
    val thresholds: DecisionThresholds,
    val tPix: Float,
    val targetItemSideFraction: Float,
    val targetItemAreaFraction: Float,
    val backgroundRgb: IntArray,
    val blurThreshold: Double,
) {
    companion object {
        fun parse(json: String): CalibratedConfig {
            val o = JSONObject(json)
            val q = o.getJSONObject("quality_and_finder")
            val cuts = o.getJSONArray("sev_cuts_area_pct")
            val bg = q.getJSONArray("background_rgb_median")
            return CalibratedConfig(
                category = o.getString("category"),
                bankSize = o.getInt("bank_size"),
                checkpoint = o.getString("checkpoint"),
                createdUtc = o.getString("created_utc"),
                splitSeed = o.getInt("split_seed"),
                thresholds = DecisionThresholds(
                    passScore = o.getDouble("t_pass").toFloat(),
                    rejectScore = o.getDouble("t_reject").toFloat(),
                    lowAreaCut = cuts.getDouble(0).toFloat(),
                    highAreaCut = cuts.getDouble(1).toFloat(),
                    confidenceScale = o.getDouble("confidence_scale").toFloat(),
                ),
                tPix = o.getDouble("t_pix").toFloat(),
                targetItemSideFraction = q.getDouble("target_item_side_fraction").toFloat(),
                targetItemAreaFraction = q.getDouble("target_item_area_fraction").toFloat(),
                backgroundRgb = IntArray(3) { bg.getDouble(it).toInt() },
                blurThreshold = q.getDouble("blur_threshold"),
            )
        }
    }
}

/**
 * Frame gate constants. Derived from MVTec pill train_good at 256 px (see docs/DECISIONS.md):
 * p99 luma 189..198, p1 luma 6..17, saturated fraction 0. Frozen before any real-item test.
 */
object FrameGate {
    const val DARK_P99_LUMA = 95.0      // half of the minimum MVTec p99 luma
    const val BRIGHT_P1_LUMA = 70.0     // ~4x the maximum MVTec p1 (background) luma
    const val GLARE_SATURATED_FRACTION = 0.005 // MVTec max is 0
    const val SATURATED_LUMA = 250.0
}

/** Item finder constants (addendum 2.2). Fixed before testing. */
object FinderConstants {
    const val AREA_BAND = 0.35f          // +-35% around the reference area
    const val MIN_SOLIDITY = 0.90f       // below: touching or overlapping items
    const val ASPECT_BAND = 0.35f        // relative aspect-ratio deviation allowed
    const val MAX_COLOUR_DISTANCE = 60.0 // RGB Euclidean distance from the reference mean colour
    const val NOISE_FRACTION_OF_REF = 0.15f // blobs smaller than this share of the reference are ignored
    const val MAX_ITEMS = 12
    const val WORK_MAX_SIDE = 1024
}
