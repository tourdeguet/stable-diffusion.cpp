# Plan da capo — sd.cpp, Rama B (convs Q8_0, VRAM primero)

Fecha: 2026-11-10. Sustituye el inventario mezclado. Alcance estricto: **sd.cpp + modelo Q8**.
Objetivo declarado del usuario: liberar VRAM de trabajo en la GTX 1060 (6 GB); la RAM/VRAM
es el mayor cuello de botella en la 1060, no el cómputo.

---

## RESULTADO (2026-11-10 15:35) — Rama B COMPLETADA Y VERIFICADA

Pipeline convs-Q8_0 en layout GEMM 2D funcionando END-TO-END, EXIT=0:

| Métrica | Rama B (Q8 convs) | Referencia F16 (safetensors) |
|---|---|---|
| Modelo en VRAM | **1,19 GiB** (params 1159,93 MB, cabe) | 3,28 GiB DiT → **NO cabe**: auto-fit manda params a CPU |
| Sampling 20 pasos 512x512 | **1,35 it/s** (14,8 s) | 0,095 it/s (210 s, pesos por PCIe cada paso) |
| generate_image total | **20,04 s** | 249,68 s |
| VAE decode | 1,88 s | — |
| Calidad vs F16 (seed 42) | SSIM 0,80 / PSNR 23,3 dB — divergencia caótica de detalles finos; ambas imágenes coherentes, sin artefactos, equivalente en calidad visual | — |

Detalle del modelo final `D:\GEMMA3\sd15-inpaint-ar181-q8b.gguf` (1,19 GiB):
convs 3x3 → Q8_0 2D `[ic*kh*kw, oc]` (419 tensores), convs 1x1 → Q8_0 (186),
ffn/attn Linear → Q8_0, F32 solo: embedding CLIP (145 MiB), time_embed, biases,
y 3 convs con fila GEMM indivisible (`ic*kh*kw % 32 != 0`).

Diferencias contra lo planeado en secciones §1-§3 (verificados durante la ejecución):

1. **El aplanado lo resuelve el EXPORTADOR, no el grafo**: `collect_tensors_for_export`
   aplane el storage 4D SOR `[oc][ic][kh][kw]` → fila GEMM `[ic*kh*kw, oc]`. El orden de
   bytes crudos NO cambia: la fila ya es `ic*(KH*KW)+kh*KW+kw`, exactamente el índice de
   patch de `ggml_im2col` (ops.cpp:7014) — sin transponer. El tensor destino en el GGUF
   guarda la FORMA original 4D (guardada en TensorExportInfo como `orig_storage`) y se
   carga como 4D; la conversión cuantizada lee por fila GEMM aplana.
2. **Divisibilidad se evalúa sobre la fila GEMM**, no sobre ne[0] del storage: la puerta
   del 73 % F32 era `tensor_should_be_converted` rechazando todo 4D por `3 % 32 != 0`
   (`model_loader.cpp:1680`). Corregida con la fila GEMM.
3. **La rama cuantizada viva está en `Conv2d::forward`** (`ggml_block.hpp:586`), no en
   `ggml_ext_conv_2d`: im2col **F32** (NO F16: el backend CUDA rechaza
   `mul_mat src1=F16 con src0 cuantizado`, ggml-cuda.cu:7669) → `mul_mat(w Q8_0, patches)`
   con peso como src0 → resultado `[OC, Nprod]` → `reshape_4d(OC,OW,OH,N)` →
   **`ggml_permute(2,0,1,3)`** (¡la semántica de permute sigue la dim ANTIGUA i a la
   posición axis_i!) → cont → add_inplace del bias.
4. **Trampa de build agravada**: con 27 TUs incluyendo `ggml_block.hpp` y
   `Conv2d::forward` inline-implícita en la clase, objetos rancios (F16 viejo) GANAN
   el link. Tras editar el header: borrar TODOS los `*.obj` de
   `CMakeFiles\stable-diffusion.dir` y relink (memoria #53, peor que lo documentado).
5. **El crash forense de la mañana (0xC0000005 en `~ModelLoader`)** era binario
   mezclado de relinks parciales, no un bug de código (memoria #53). El handler
   `[P61-CRASH]` de main.cpp (RIP/RSP + stack simbolizado contra sd-cli.map) se
   CONSERVA — solo se dispara en fallo. ASAN: descartado (nor funciona en esta
   máquina, exe min falsea 0xC0000142).
6. Debug retirado: `[P61-T]`/`[P61-DTOR]`/`[P61-DEBUG]`. Quedan: `[P61-SCHED]`
   (ggml-backend.cpp, solo cuando un op queda sin asignar) y contador de estado
   `[P61] export mix` en convert.cpp.

Pendiente (siguiente turno): img2img/inpaint con el modelo Q8 (las convs cuantizadas
sirven igual); ir2img latents path NO verificado; commit del parche.

---

## 0. Estado de partida (verificado en disco, 2026-11-10)

| | |
|---|---|
| Repo | `D:\sd-cpp`, HEAD `e239e4b`, parche sin commitear (`src/convert.cpp`, `src/model_loader.cpp/.h`, `src/model/diffusion/unet.hpp`, submódulo `ggml`) |
| Binario | `build-p61-baseline\bin\sd-cli.exe` (10-nov 01:53), compila y arranca |
| GGUF existente | `D:\GEMMA3\sd15-inpaint-ar181-q8_0.gguf` (3,0 GiB) — **corrupto, NO medir con él** |
| Comprobación | el fichero analizado declara Q8_0 pero contiene F32 (bug de upstream, ver §1) |

### Hechos que sobreviven al da capo

1. **Bug del conversor (upstream, confirmado byte a byte):** `load_tensor_for_export`
   (`convert.cpp`) crea el tensor destino ya en el tipo final y llama
   `ModelLoader::load_tensor` (`model_loader.cpp:1448`), que **copia bytes crudos sin
   convertir** → un bloque Q8_0 recibe F32. Afecta a todo GGUF hecho con
   `sd-cli -M convert --type q8_0` en este HEAD. Con el parche actual, el camine de
   exportación ya carga en tipo origen + `convert_tensor` (funciona).
2. **Por qué las convs quedan F32:** `Conv2d::init_params`
   (`src/model/common/ggml_block.hpp:519`) fuerza `wtype = GGML_TYPE_F16` e **ignora
   `expected_type`**. Además, el peso 4D `[kw(3), kh(3), ic, oc]` tiene `ne[0] = 3`:
   Q8_0 exige bloques de 32 contiguos → **no es cuantizable en este layout**.
   Cuantizable solo tras aplanar a la forma GEMM `[ic*kh*kw, oc]` (5760 para 640 ch).
3. **El kernel P3 (implicit-GEMM) no tiene bug de layout** (doc R6, §7.1): indexa
   OC-major/WHCN igual que `whcn_layout`. Su único relieve es stride/dilation>1,
   irrelevante para la UNet SD1.5. NO es el cuello: en producción sin parche y
   `sd_probe_conv_direct` apaga la ruta directa en sm_61
   (`ggml_extend_backend.cpp:594` exige `arch >= 750`) → la ruta real es
   **im2col + mul_mat (MMQ)**. Con convs en Q8_0 esa ruta es exactamente la
del Linear ya cuantizado.
4. **MMVQ Pascal en sd.cpp: NO propagar.** En difusión `ne11` es 64-4096 y MMVQ solo
   entra con `ne11<=8` (prácticamente solo `time_embed`). Bajo rendimiento esperado.
5. **Nota de fusión (senior, Fase 3.1):** si después del volcado vamos a
   cuantizar activaciones a `q8_1` inmediatamente tras im2col
   (`im2col_q8_1_cuda`), esto es la fase 2/3 del plan del senior, NO de hoy.
   El today-plan (Rama B) no depende de eso: `ggml_conv_2d` ya funciona con
   im2col → mul_mat y cuantización de activaciones al vuelo (MMQ).
6. **El imatrix de sd.cpp ya existe** (`src/runtime/imatrix.cpp`, CLI `--imatrix`),
   pero solo cubre `GGML_OP_MUL_MAT`. Y `quantize_q8_0` la ignora (Q8_0 no usa
   imatrix). Solo aplicable a Q6_K/Q4_K, fase posterior.

Cómo se llegó: nota de sesión #4 + memoria #52 + informe
`ENTREGA-senior-CONVERSION-Q8.md` §7.

---

## 1. Qué atacar (Rama B)

### Paso 1 — Conversor: exportar convs como 2D GEMM-layout Q8_0

- En `convert.cpp`, en la recolección de tensores a exportar (o en
  `unet.hpp`/`ggml_block.hpp`, según convenga), los `.weight` **4D de Conv2d** se
  aplanan a **2D `[ic*kh*kw, oc]`** antes del GGUF.
- Aplicar `--type q8_0` solo a esos tensores (regla de `--tensor-type`
  `"*\w*"` 4D o equivalente). **No tocar** time_embed, bias, proj_in 1x1,
  CLIP/VAE salvo decisión explícita.
- Guarda: si `in_channels == 1` o `kernel_size > 3x3` (u otras formas raras),
  probar divisibilidad de la fila GEMM alfa `in_channels * kh * kw`:
  solo si `(ic*kh*kw) % grid == 0` se exporta Q8_0; si no, F16.

### Paso 2 — Grafo: Conv2d::forward con peso cuantizado → im2col + mul_mat

- En `Conv2d::forward` (`ggml_block.hpp:551`): si `w->type` es cuantizado
  (`ggml_is_quantized`), construir el grafo como el patrón del Linear:
  `im2col(x)` + `ggml_mul_mat` contra el peso 2D cuantizado.
  Si no, mantener `ggml_ext_conv_2d` de siempre (camino F16/F32).
- La fusión `im2col_q8_1` (Fase 3.1 del senior) queda para después; no es
  pre-requisito de Rama B.

### Paso 3 — Rebuild y reconversión

1. Borrar `.obj` de `sd.cpp` antes de reconstruir (ninja no recompila al tocar
   `ggml_block.hpp`, es header de `src/model/integrations/sd.cpp`).
2. `D:\vid3d\build-sd-p61.bat build`.
3. Reconvertir el checkpoint a GGUF: `sd-cli -M convert -m ... -o D:\GEMMA3\out.gguf
   --type q8_0`. Convertir SIEMPRE a `D:` (en `C:`, solo quedan 0,7 GB libres y
   el conversor preasigna).
4. Verificar tipos con `sd_gguf_types.py`: convs 3x3 → Q8_0, resto no tocado.

### Paso 4 — Verificar que el fichero NO está corrupto

- Leer bytes crudos de un bloque Q8_0 aleatorio de una conv (no solo el header):
  valores de `d` Razonables en rango F32, resto de bytes coherentes con el tipo.
  Complementa `sd_gguf_types.py`, que solo mira la cabecera.
- Inferencia 512x512x20 pasos, seed fija: comprobar que no hay `NaN` ni
  imagen corrupta. Si hay NaN, el fichero está mal.

### Paso 5 — Medición end-to-end

- F16 stock (checkpoint safetensors) vs modelo Q8. Métricas: s/it y SSIM.
  No espere ganancia grande de cómputo: Q8 vs F16 en cómputo es par en la
  1060 (DP4A). La ganancia real de Rama B es **VRAM liberada**: que el grafo
  que NO cupo en los 6 GB pase a cupir.

---

## 2. Qué NO hacer (listado explícito para no resucitar)

- **NO propagar** el parche de MMVQ de llama-pascal61 a sd.cpp.
- **NO medir** con el GGUF actual de `D:\GEMMA3\sd15-inpaint-ar181-q8_0.gguf`
  (corrupto por el bug del conversor).
- **NO usar el modo memoria** (`convert_model_in_memory`) para exportar
  con override de tipo — con el override aborta; con exportación por streaming va bien.
- **NO baseline** contra `sd_f0_unet_bench.py` (192 tensores missing, bug de
  arquitectura). Para cifras de UNet estático solo sirve `sd_f0_baseline.py`.
- **NO tocar** `Conv2d::init_params` para llamar `get_type`: el problema no es
  el tipo de `init_params`, es la forma 4D con `ne[0]=3`. (Lo que se escribió
  en el R6 de `ENTREGA-senior-CONVERSION-Q8.md` §7.2 está DESACTUALIZADO:
  la "corrección de 1 línea" no es aplicable tal cual; ver §1 de este plan.)

---

## 3. Desviaciones respecto a los documentos anteriores

- `ENTREGA-senior-CONVERSION-Q8.md` §7.2 afirmaba que cuantizar las convs es
  "un cambio de 1 línea" en `init_params`. Revisión: falta el layout GEMM,
  que no es 1 línea; ver §1 de este plan.
- Se descartan del plan del senior las fases 2/3 (MMQ tiling, mmq-config
  `pascal-dp4a`, im2col-q8_1, flash-attn), que quedan como fases posteriores
  NUEVAS, no como parte de la Rama B.
- Ya existe `imq-config-pascal-dp4a.cuh` TAG_MMQ: disponible si hace falta;
  no es bloqueante de Rama B.
