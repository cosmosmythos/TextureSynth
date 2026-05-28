@echo off
setlocal

REM Force Khronos validation + sync validation + best-practices
set VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
set VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT;VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT

REM Crash on first validation error (do NOT use in production)
set VK_LAYER_PRINTF_TO_STDOUT=1

echo === Running engine_tests with full validation ===
"%~dp0..\build\tests\Release\engine_tests.exe" %*
endlocal
