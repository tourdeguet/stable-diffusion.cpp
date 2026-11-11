# P3 aplicado y medido — Conv2D en Pascal (sm_61)

Fecha: 2026-10-31. Estado: **P3 medido y confirmado**, listo para integrar.

## Qué se hizo

`ggml/src/ggml-cuda/conv2d.cu`, en `ggml_cuda_op_conv2d`. Se habilitó la ruta
implicit-GEMM para NVIDIA sin tensor cores:

```cpp
// Antes
const bool use_simt = GGML_CUDA_CC_IS_MTHREADS(device.cc) ||
                      (GGML_CUDA_CC_IS_AMD(device.cc) && !use_mma && IC * KW * KH >= 64);

// Después (+ condición NVIDIA, con `pointwise`/`use_blas` declarados antes)
const bool pointwise = KW == 1 && KH == 1 && ST_X == 1 && ST_Y == 1 && PD_X == 0 && PD_Y == 0;
const bool use_blas  = pointwise && fast_fp16_hardware_available(device.cc);
const bool use_simt  = GGML_CUDA_CC_IS_MTHREADS(device.cc) ||
                       (GGML_CUDA_CC_IS_AMD(device.cc) && !use_mma && IC * KW * KH >= 64) ||
                       (GGML_CUDA_CC_IS_NVIDIA(device.cc) && !use_mma && !use_blas && IC * KW * KH >= 64);
```

El parche completo está en `D:\vid3d\pascal-conv2d.patch`.

## Corrección al diagnóstico previo del plan v4

El plan decía «en sm_61 cae siempre al kernel SIMT genérico». **No es exacto.**
`use_simt` no tenía rama NVIDIA, así que en sm_61 no se cumplía ninguna de las
tres condiciones (`use_mma`, `use_blas`, `use_simt`) y se caía al kernel
**naive** `conv2d_cuda_f16`, que no reutiliza datos.

## Medición

Micro-banco `D:\sd-cpp\tests\conv2d_bench.cpp`: conduce `GGML_OP_CONV_2D` por la
API pública de ggml, que despacha a la misma `ggml_cuda_op_conv2d` que usa sd.cpp.

Entorno: GTX 1060 6GB (sm_61), CUDA 11.8, MSVC 14.29, `-DCMAKE_CUDA_ARCHITECTURES=61-real`, Release.
Mediana de 7 ejecuciones, 3 de calentamiento.

| Caso | Naive | Implicit-GEMM | Ganancia |
|---|---|---|---|
| 3x3 32->32   720p (1280x720) | 263 ms | **22,3 ms** | 11,8x |
| 3x3 32->64   720p | 527 ms | **22,0 ms** | 24,0x |
| 3x3 64->64   720p | 1050 ms | **40,4 ms** | 26,0x |
| 3x3 128->128 720p | 4191 ms | **147,8 ms** | 28,4x |
| 3x3 256->256 720p | 16924 ms | **566,1 ms** | 29,9x |
| 3x3 64->64   1080p (1920x1080) | 2417 ms | **90,0 ms** | 26,9x |
| 1x1 64->64   720p | 297 ms | **10,2 ms** | 29,1x |

Lectura: 32->32 y 32->64 cuestan lo mismo (22 ms), señal de que el coste pasa a
estar dominado por el tiling del tile y no por `IC*OC`. El escalado anterior era
lineal en `IC*OC`, firma del kernel naive.

## Cómo reproducir

```bat
:: build (ya configurado en D:\sd-cpp\build-p61-baseline)
D:\vid3d\build-sd-p61.bat build

:: el banco
D:\vid3d\build-conv2d-bench.bat
set PATH=D:\sd-cpp\build-p61-baseline\bin;%PATH%
D:\sd-cpp\build-p61-baseline\bin\conv2d_bench.exe
```

## Trampa encontrada (importante para medir)

El `.lib` estático de ggml-cuda **no se re-enlaza** en el ejecutable del banco si
solo recompilas ggml. La primera medición «del fork» dio ganancia cero porque el
`conv2d_bench.exe` seguía enlazado contra la librería anterior (mtime del `.exe`
anterior al del `.lib`). **Recompilar el banco después de cada build de ggml.**

## Nota de toolchain

El script de build imprimía `mt.exe: general error c101008d` dentro de un
`echo` con `...` y `cmd` lo interpretaba como comando, devolviendo exit 255 con el
build correcto. Sustituido por un mensaje sin puntos suspensivos, pero el exit
code no afecta al resultado.

## Qué queda

- P4 (CUDA graphs) sigue pendiente y no se ha tocado.
- P2 (conv cuantizada nativa) sigue sin sentido hasta que P3 no sea el cuello:
  con P3 el kernel ya reutiliza datos.
- Para SD: medir el efecto real en una generación completa, no solo en el
  micro-banco. Se espera ganancia en el UNet, que es convolucional, y ninguna en
  el bloque de atención.
