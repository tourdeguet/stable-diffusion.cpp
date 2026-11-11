# Plan v4 — Cuantización dinámica con imatrix + fork de stable-diffusion.cpp

Fecha: 2026-10-30. Reemplaza al v3. Las correcciones vienen de leer el código, no de suponerlo.

## 0. El cuantizador ya existe y es reutilizable

`D:\SD-UI\tools\quantize_sdxl_gguf.py` llama a **`ggml_quantize_chunk` de `ggml-base.dll`**
(ctypes) — el kernel real de ggml, no una reimplementación. `gguf-py` disponible en
`D:\llama.cpp\gguf-py`. Pasa `None` en el parámetro `imatrix` (línea 51).

**Añadir imatrix = cambiar ese `None` por un puntero.** No hay cuantizador que escribir.

## 1. Corrección importante: la imatrix no ayuda en las convoluciones (Q8_0)

Leído en `ggml/src/ggml-quants.c`:

```c
size_t quantize_q8_0(...) {
    (void)quant_weights;   // not used
    ...
}
```

**Q8_0 ignora la imatrix por completo.** Sólo Q6_K (y Q4_0, Q4_K, etc.) la usan:

```c
size_t quantize_q6_K(...) {
    if (!quant_weights) quantize_row_q6_K_ref(...);
    else quantize_row_q6_K_impl(src, qrow, n_per_row, quant_weights);   // usa pesos
}
```

Consecuencia sobre la estrategia pedida (**Q8 convs + Q6 atención/FFN con imatrix**):

| Parte | Tipo | ¿La imatrix aporta? |
|---|---|---|
| Convs 3x3 y 1x1 | Q8_0 | **No.** A 8 bits el error es tan bajo que no hay nada que repartir |
| Atención y FFN | Q6_K | **Sí**, y es donde importa |

No es un problema: la imatrix se aplica exactamente donde el error es no nulo. Pero conviene
saberlo antes de gastar días generándola para las convs.

## 2. El bloqueante real: una conv cuantizada no se puede ejecutar hoy

`ggml/src/ggml-cuda/conv2d.cu:347` (sd.cpp):

```cpp
GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
```

Y en Forge, `backend/operations_gguf.py` dequantiza el tensor a `computation_dtype` antes de
la operación. Así que con la ruta actual:

**Q8_0 en convs = ahorro de disco únicamente.** Ni VRAM ni velocidad.

Para que la cuantización de convs aporte de verdad hace falta **una conv cuantizada nativa**
en sd.cpp: leer bloques Q8_0/Q6_K y multiplicarlos directamente en el tile. Es el trabajo de
fork real, y es el equivalente exacto de lo que hicimos con MMVQ.

## 3. Estado del fork de sd.cpp

Clonado en **`D:\sd-cpp`** (HEAD `e239e4b`, submódulos init, ggml de `leejet/ggml`, 75 `.cu`).

Lo que ya viene heredado y ahorra trabajo:

- **`mmq-config-pascal-dp4a.cuh` existe** → mismo parche de Pascal que en nuestro llama.cpp.
- `calc_nwarps()` en `mmvq.cu:452` es el **mismo punto de inserción** que en p61:
  hoy sólo conoce GENERIC, TURING, GCN, RDNA2/3/4, GB10. **`MMVQ_PARAMETERS_PASCAL`
  se traslada tal cual** (Q4_K nwarps=1, Q6_K/Q8_0 nwarps=2, rows_per_block=2).
- `conv2d.cu:379-386`: la selección de ruta depende de `use_mma` (turing+), `use_blas`
  (fast fp16) y `use_simt` (sólo AMD/MThreads). **En sm_61 cae siempre al kernel SIMT
  genérico**, sin im2col implícito ni split-K afinado. Ahí está el margen.

## 4. Palancas, por orden de valor medido

| # | Cambio | Origen | Dónde toca |
|---|---|---|---|
| **P1** | `MMVQ_PARAMETERS_PASCAL` + `rows_per_block=2` | medido en p61: Q4_K +13 %, Q8_0 +41 % | `mmvq.cu:452` |
| **P2** | Conv cuantizada Q8_0/Q6_K en sm_61 | nuevo | `conv2d.cu` (kernel nuevo) |
| **P3** | Ruta `conv2d` con im2col en shared memory para sm_61 (aunque sea F16) | nuevo | `conv2d.cu:392` (habilitar `use_simt` para NVIDIA) |
| **P4** | CUDA graphs para el grafo de la UNet | parche #27721 | runner |

P3 es el atajo de P2: si se habilita la ruta im2col para sm_61 **con pesos F16**, se gana
velocidad sin tocar la cuantización. Es una línea de condición y hay que medirla antes que
nada, porque si rinde, P2 pasa a ser "P3 + dequant en el tile".

## 5. Orden de ejecución

```
F0  baseline SD-UI (cuDNN al PATH, 512x512, 20 steps, seed fija)        30 min
F1  fp16 + VAE f32 (--no-half-vae)                                      30 min
F2  build sd.cpp sm_61 + medir stock vs SD-UI                           medio día
F3  P1: MMVQ Pascal                                                    1 día
F4  P3: im2col F16 en sm_61 (una condición)                            1 día
F5  imatrix para SD (hooks en el UNet de Forge)                        1-2 días
F6  variante Q8convs+Q6attn con imatrix (extiende el script)           1 día
F7  P2: conv cuantizada nativa                                         3-5 días
```

Antes de F5 conviene saber si F4 ya da la mayor parte de la ganancia: si el prefill de
difusión con im2col F16 rinde, la cuantización de convs pasa a ser optimización de memoria
(que sí importa para caber a 768x768), no de velocidad.

## 6. Detalle de la imatrix para difusión

`llama-imatrix` mide columnas con corpus de texto. En difusión la magnitud análoga es la
**media cuadrática de las activaciones de entrada por columna**, muestreada en varios
timesteps (las activaciones cambian mucho con t).

Formato esperado por `ggml_quantize_chunk`: `const float * imatrix`, **una entrada por
columna de entrada del tensor** (`n_per_row`), idéntico para todas las filas.

Plan: hook en el `UNetModel` de Forge que corra 5-10 steps sobre N pares (imagen, máscara,
prompt) del dataset de inpainting, acumulando `sum(x²)` por columna de cada
`attn*.to_*.weight` y `ff.net.*.weight`. Salida `.npz`. Se pasa al script de cuantización
como `--imatrix archivo.npz`, y éste lo entrega a `ggml_quantize_chunk`.

Sólo se generan para los tensores Q6_K (atención y FFN).

## 7. Criterio de éxito

| Métrica | Objetivo |
|---|---|
| GGUF SD1.5 (Q8 convs + Q6 attn/FFN) | ≤ 1,2 GiB (vs 3,97 f32 / 1,99 f16) |
| SSIM vs f32, seed fija | ≥ 0,97 |
| s/it SD-UI con fp16 | −20 % vs f32 |
| s/it sd.cpp fork (P1+P3) | medir contra stock de sd.cpp el mismo día |

## 8. Riesgos

1. **Q8_0 ignora la imatrix** (confirmado en código). No esperar mejora de calidad por ella
   en las convs.
2. **Conv cuantizada no ejecutable hoy**: sin P2, la cuantización de convs sólo ahorra disco.
3. **VAE**: nunca cuantizar, ni siquiera Q8. F16 con `--no-half-vae`.
4. **Checkpoints XL presentes** en `models\Stable-diffusion` (UNet 2.6 B): no mezclar sus
   medidas con las de SD1.5.
5. `cudnn 8.9.7` en `D:\` es de **cuda11** y torch es **cu121**: verificar
   `torch.backends.cudnn.version()` antes de medir.
