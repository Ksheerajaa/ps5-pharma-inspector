package com.example.ps5.advisory

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.roundToInt

/**
 * The single JSON hand-off between the vision pipeline and the advisory pipeline (FAQ 4 Q5):
 * defect, severity, confidence, box, plus verdict and location. Llama never sees pixels.
 */
data class AdvisoryRecord(
    val product: String,
    val itemNo: Int,
    val verdict: String,
    val defectType: String,
    val severity: String,
    val confidencePct: Int,
    val areaPct: Double,
    val location: String,
    val box: IntArray?,
    val recentRejectsLast20: Int,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("product", product)
        put("item", itemNo)
        put("verdict", verdict)
        put("defect_type", defectType)
        put("severity", severity)
        put("confidence_pct", confidencePct)
        put("area_pct", areaPctRounded())
        put("location", location)
        put("box", box?.let { JSONArray(it.toList()) } ?: JSONObject.NULL)
        put("recent_rejects_last_20", recentRejectsLast20)
    }

    fun areaPctRounded(): Double = (areaPct * 10).roundToInt() / 10.0

    /** Every number a valid answer may mention. */
    fun allowedNumbers(): Set<String> {
        val values = mutableListOf(
            itemNo.toDouble(), confidencePct.toDouble(), areaPct, recentRejectsLast20.toDouble(), 20.0,
        )
        box?.forEach { values.add(it.toDouble()) }
        return values.flatMap { numberForms(it) }.toSet()
    }

    companion object {
        fun numberForms(v: Double): List<String> = listOf(
            v.roundToInt().toString(),
            "%.1f".format(java.util.Locale.ROOT, v),
            "%.2f".format(java.util.Locale.ROOT, v),
        ).flatMap { listOf(it, it.trimEnd('0').trimEnd('.')) }
    }
}

object SopHints {
    // Illustrative engineering guidance only. Not GMP or regulatory advice.
    private val hints = mapOf(
        "crack or scratch" to "Check punch and die wear and tablet hardness; inspect neighbouring tablets.",
        "contamination or colour fault" to "Check for foreign particles, lubricant spots and blend uniformity.",
        "deformation" to "Check compression force and tooling alignment.",
        "unspecified anomaly" to "Inspect the marked area manually and record what you see.",
        "none" to "No defect detected.",
    )
    fun hintFor(type: String) = hints[type] ?: hints.getValue("unspecified anomaly")
}

object PromptBuilder {
    /**
     * Prompt v2 (2026-11-23). v1 measured 26/50 valid (seed 2026): the model repeated the block, replaced the
     * verdict verb ("Review"), exceeded 20 words, and paraphrased "the verdict was already decided" as a reason.
     * v2: shorter target (15 words), one worked example, and the ACTION line is pre-filled with the verb that the
     * decision core dictates, so Llama explains a decision it cannot change.
     */
    const val PROMPT_VERSION = "v2"

    const val SYSTEM = """You write short notes for operators on a tablet inspection line.
Write exactly four lines and nothing else:
ACTION: finish the given sentence about what the operator does with this item.
REASON: the defect type, severity and location from the record.
CHECK: what to check on the line, based on the SOP hint.
LOG: one short sentence for the batch record.
Each line has at most 15 words. Use only facts from the record and the SOP hint.
Do not write any number except the item number. Plain text, no markdown."""

    private const val EXAMPLE_USER = """product: pill
item: 7
verdict: REJECT
defect_type: deformation
severity: HIGH
area_pct: 2.4
location: left
confidence_pct: 91
recent_rejects_last_20: 1
SOP_HINT: Check compression force and tooling alignment."""

    private const val EXAMPLE_ANSWER = """ACTION: Reject item 7 and keep it apart from good tablets.
REASON: High severity deformation on the left side of the tablet.
CHECK: Check compression force and tooling alignment on the press.
LOG: Item 7 rejected for high severity deformation."""

    /** Verb fixed by the decision core; the model continues the sentence. */
    fun prefill(r: AdvisoryRecord): String {
        val verb = when (r.verdict) {
            "REJECT" -> "Reject"
            "RE-INSPECT" -> "Re-inspect"
            else -> "Release"
        }
        return "ACTION: $verb item ${r.itemNo}"
    }

    /** Pre-fill + generated continuation, cut after the first complete LOG line (later text is discarded). */
    fun completeAnswer(r: AdvisoryRecord, generated: String): String {
        val full = prefill(r) + generated.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
        val m = LOG_LINE.find(full) ?: return full
        return full.substring(0, m.range.last + 1).trimEnd()
    }

    /** True once the generated text contains a finished LOG line; used to stop generation early. */
    fun isComplete(r: AdvisoryRecord, generated: String): Boolean = LOG_LINE.containsMatchIn(prefill(r) + generated)

    private val LOG_LINE = Regex("""(?m)^LOG:[^\n]*\S[^\n]*(?=\n)""")

    fun userMessage(r: AdvisoryRecord): String = buildString {
        appendLine("product: ${r.product}")
        appendLine("item: ${r.itemNo}")
        appendLine("verdict: ${r.verdict}")
        appendLine("defect_type: ${r.defectType}")
        appendLine("severity: ${r.severity}")
        appendLine("area_pct: ${r.areaPctRounded()}")
        appendLine("location: ${r.location}")
        appendLine("confidence_pct: ${r.confidencePct}")
        appendLine("recent_rejects_last_20: ${r.recentRejectsLast20}")
        append("SOP_HINT: ${SopHints.hintFor(r.defectType)}")
    }

    /** Llama 3 chat template. The runner is configured with numBos = 0, so BOS is written here once. */
    fun llama3Prompt(r: AdvisoryRecord): String =
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n$SYSTEM<|eot_id|>" +
            "<|start_header_id|>user<|end_header_id|>\n\n$EXAMPLE_USER<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n$EXAMPLE_ANSWER<|eot_id|>" +
            "<|start_header_id|>user<|end_header_id|>\n\n${userMessage(r)}<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n${prefill(r)}"
}

data class AdvisoryText(
    val action: String,
    val reason: String,
    val check: String,
    val log: String,
) {
    fun asLines() = "ACTION: $action\nREASON: $reason\nCHECK: $check\nLOG: $log"
}

object AdvisoryValidator {
    val LABELS = listOf("ACTION", "REASON", "CHECK", "LOG")
    const val MAX_WORDS = 20
    private val keywords = mapOf(
        "REJECT" to listOf("reject", "remove", "segregate", "quarantine"),
        "RE-INSPECT" to listOf("re-inspect", "reinspect", "recheck", "re-check", "inspect again"),
        "PASS" to listOf("release", "continue", "pass"),
    )
    private val numberRegex = Regex("""\d+(?:\.\d+)?""")
    private val OUTCOME_CLAIM = Regex("""\b(found|no (issues?|defects?|problems?|faults?)|confirmed|cleared|passed|released)\b""")

    sealed interface Result {
        data class Valid(val text: AdvisoryText) : Result
        data class Invalid(val reason: String) : Result
    }

    fun validate(raw: String, record: AdvisoryRecord): Result {
        val cleaned = raw.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").trim()
        val lines = cleaned.lines().map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.size != 4) return Result.Invalid("expected 4 lines, got ${lines.size}")
        val values = mutableListOf<String>()
        for ((line, label) in lines.zip(LABELS)) {
            val prefix = "$label:"
            if (!line.uppercase().startsWith(prefix)) return Result.Invalid("line must start with $prefix")
            val body = line.substring(prefix.length).trim()
            if (body.isEmpty()) return Result.Invalid("$label is empty")
            if (body.split(Regex("\\s+")).size > MAX_WORDS) return Result.Invalid("$label exceeds $MAX_WORDS words")
            if (body.contains('*') || body.contains('#')) return Result.Invalid("markdown in $label")
            values.add(body)
        }
        val action = values[0].lowercase()
        val own = keywords[record.verdict] ?: return Result.Invalid("unknown verdict")
        // "re-inspect" contains no other set's keyword, but "pass"/"reject" are checked on word boundaries.
        fun has(word: String) = Regex("(?<![a-z-])${Regex.escape(word)}(?![a-z])").containsMatchIn(action)
        if (own.none { has(it) }) return Result.Invalid("ACTION does not match verdict ${record.verdict}")
        for ((verdict, words) in keywords) {
            if (verdict == record.verdict) continue
            val conflicting = words.filter { has(it) && own.none { o -> o.contains(it) } }
            if (conflicting.isNotEmpty()) return Result.Invalid("ACTION contains conflicting keyword ${conflicting.first()}")
        }
        // Added after manual review of the seed-2027 run (3/50 RE-INSPECT answers invented the re-inspection result,
        // e.g. "found no issues"): a pending re-inspection has no outcome yet, so any claimed outcome is rejected.
        if (record.verdict == "RE-INSPECT") {
            val claim = OUTCOME_CLAIM.find(values.drop(1).joinToString(" ").lowercase())
            if (claim != null) return Result.Invalid("claims re-inspection outcome '${claim.value}'")
        }
        val allowed = record.allowedNumbers()
        for (m in numberRegex.findAll(values.joinToString(" "))) {
            if (m.value !in allowed) return Result.Invalid("invented number ${m.value}")
        }
        return Result.Valid(AdvisoryText(values[0], values[1], values[2], values[3]))
    }
}

object AdvisoryTemplates {
    fun forRecord(r: AdvisoryRecord): AdvisoryText {
        val where = if (r.location == "none") "" else " at ${r.location}"
        return when (r.verdict) {
            "REJECT" -> AdvisoryText(
                action = "Reject item ${r.itemNo} and segregate it from the batch.",
                reason = "${r.severity.lowercase()} ${r.defectType}$where.",
                check = SopHints.hintFor(r.defectType),
                log = "Item ${r.itemNo} rejected: ${r.defectType}, ${r.severity.lowercase()} severity.",
            )
            "RE-INSPECT" -> AdvisoryText(
                action = "Re-inspect item ${r.itemNo} manually; hold it until checked.",
                reason = "Possible ${r.defectType}$where; score between thresholds.",
                check = SopHints.hintFor(r.defectType),
                log = "Item ${r.itemNo} held for re-inspection: possible ${r.defectType}.",
            )
            else -> AdvisoryText(
                action = "Release item ${r.itemNo}; continue inspection.",
                reason = "No anomaly above the calibrated threshold.",
                check = "No action needed.",
                log = "Item ${r.itemNo} passed.",
            )
        }
    }
}
