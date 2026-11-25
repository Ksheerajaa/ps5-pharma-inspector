package com.example.ps5.ui

import kotlinx.coroutines.delay

/**
 * Scripted walkthrough for recording the demo video without touching the phone:
 *   adb shell am start -n com.example.ps5/.MainActivity --ez demo true
 * Timings match docs/DEMO_SCRIPT.md. Uses only replay images already on the phone.
 */
object DemoScript {
    suspend fun run(setTab: (Int) -> Unit, showReplay: (String?) -> Unit) {
        setTab(0); delay(8_000)                                         // Inspect: live camera, header
        setTab(1); delay(1_500)
        showReplay("testcases/I3_trays/tray2x2.png"); delay(30_000)    // 4-item tray, reject + Llama advice
        showReplay("testcases/I9_mixed_cluster/frame.png"); delay(12_000) // touching items -> not inspected
        showReplay("testcases/F4_glare/frame.png"); delay(9_000)        // quality gate
        showReplay("testcases/I1_good/frame.png"); delay(9_000)         // all pass
        setTab(2); delay(14_000)                                        // History log
        setTab(3); delay(9_000)                                         // Bench
        setTab(4); delay(12_000)                                        // About / offline / Built with Llama
        setTab(0)
    }
}
