package com.example.ps5.advisory

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AdvisoryTest {
    private val reject = AdvisoryRecord("pill", 3, "REJECT", "crack or scratch", "HIGH", 82, 2.43, "upper-left", null, 1)
    private val reinspect = reject.copy(verdict = "RE-INSPECT", severity = "LOW")

    private fun valid(raw: String, r: AdvisoryRecord) = AdvisoryValidator.validate(raw, r) is AdvisoryValidator.Result.Valid
    private fun reason(raw: String, r: AdvisoryRecord) = (AdvisoryValidator.validate(raw, r) as AdvisoryValidator.Result.Invalid).reason

    private val good = """ACTION: Reject item 3 and segregate it.
REASON: High severity crack or scratch at upper-left, confidence 82%.
CHECK: Check punch and die wear and tablet hardness.
LOG: Item 3 rejected for crack, area 2.4 percent.<|eot_id|>"""

    @Test
    fun acceptsWellFormedOutput() {
        assertTrue(valid(good, reject))
    }

    @Test
    fun a1RejectsMalformedOutput() {
        assertTrue(reason("Sure! Here is my answer.", reject).startsWith("expected 4 lines"))
        assertTrue(reason(good.replace("CHECK:", "NOTE:"), reject).contains("CHECK"))
    }

    @Test
    fun a4RejectsInventedNumber() {
        assertEquals("invented number 47", reason(good.replace("confidence 82%", "confidence 47%"), reject))
    }

    @Test
    fun a5RejectsActionContradictingVerdict() {
        assertTrue(reason(good.replace("Reject item 3 and segregate it.", "Release item 3 to packing."), reject).contains("does not match"))
        val mixed = good.replace("Reject item 3 and segregate it.", "Re-inspect item 3, then release.")
        assertTrue(reason(mixed, reinspect).contains("conflicting"))
    }

    @Test
    fun reinspectMustNotClaimOutcome() {
        val r = reinspect.copy(defectType = "deformation")
        val base = "ACTION: Re-inspect item 3 and check tooling.\nREASON: Low severity deformation at upper-left.\nCHECK: Check compression force.\nLOG: "
        assertTrue(valid(base + "Item 3 held for re-inspection.", r))
        assertTrue(reason(base + "Item 3 re-inspected and found no issues.", r).contains("outcome"))
        assertTrue(reason(base + "Item 3 re-inspected and found to be defective.", r).contains("outcome"))
    }

    @Test
    fun rejectsTooManyWords() {
        val long = good.replace("Check punch and die wear and tablet hardness.", (1..25).joinToString(" ") { "word" })
        assertTrue(reason(long, reject).contains("exceeds"))
    }

    @Test
    fun promptStartsWithSingleBosAndContainsRecord() {
        val p = PromptBuilder.llama3Prompt(reject)
        assertEquals(1, p.split("<|begin_of_text|>").size - 1)
        assertTrue(p.contains("verdict: REJECT"))
        assertTrue(p.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\nACTION: Reject item 3"))
        assertEquals("ACTION: Re-inspect item 3", PromptBuilder.prefill(reinspect))
    }

    @Test
    fun completeAnswerKeepsOnlyFirstBlock() {
        val gen = " and segregate it.\nREASON: High severity crack or scratch at upper-left.\nCHECK: Check punch and die wear.\n" +
            "LOG: Item 3 rejected for crack.\n\nACTION: Reject item 3 again\nREASON: repeated"
        assertTrue(PromptBuilder.isComplete(reject, gen))
        val answer = PromptBuilder.completeAnswer(reject, gen)
        assertTrue(answer.endsWith("LOG: Item 3 rejected for crack."))
        assertTrue(valid(answer, reject))
    }

    @Test
    fun incompleteWhileLogLineIsOpen() {
        assertTrue(!PromptBuilder.isComplete(reject, " and segregate it.\nREASON: x\nCHECK: y\nLOG: Item 3 rej"))
        // Model ends with EOS directly after the LOG text (no newline): answer is still used whole.
        val answer = PromptBuilder.completeAnswer(reject, " and segregate it.\nREASON: Crack.\nCHECK: Check punch.\nLOG: Item 3 rejected.<|eot_id|>")
        assertTrue(valid(answer, reject))
    }

    @Test
    fun templatesMatchVerdict() {
        for (r in listOf(reject, reinspect, reject.copy(verdict = "PASS"))) {
            val t = AdvisoryTemplates.forRecord(r)
            assertTrue("template for ${r.verdict} must validate", valid(t.asLines(), r))
        }
    }
}
