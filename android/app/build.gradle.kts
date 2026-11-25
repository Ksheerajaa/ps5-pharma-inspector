plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.example.ps5"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.example.ps5"
        minSdk = 29
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    packaging {
        // OpenCV and fbjni (ExecuTorch) both ship the NDK libc++_shared.so.
        jniLibs { pickFirsts += "lib/**/libc++_shared.so" }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2026.06.00"))
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.11.1")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("androidx.camera:camera-camera2:1.7.0")
    implementation("androidx.camera:camera-lifecycle:1.7.0")
    implementation("androidx.camera:camera-view:1.7.0")

    implementation("androidx.room:room-runtime:2.8.5")
    implementation("androidx.room:room-ktx:2.8.5")
    ksp("androidx.room:room-compiler:2.8.5")

    implementation("org.opencv:opencv:4.13.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
    implementation("org.pytorch:executorch-android:1.5.1")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
}
