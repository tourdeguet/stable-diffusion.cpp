# Entrega al senior — fork de sd.cpp, estado real 2026-11-09/10

Verificado en disco y con git al escribir esto. Objetivo del fork: **modelo Q8**.

---

## 1. Resumen ejecutivo

| | |
|---|---|
| Rama | `D:\sd-cpp`, HEAD `e239e4b` (sin commitear) |
| Ficheros modificados | `src/convert.cpp`, `src/model_loader.cpp`, `src/model_loader.h`, `src/model/diffusion/unet.hpp`; submódulo `ggml` modificado |
| Build | `D:\sd-cpp\build-p61-baseline\bin\sd-cli.exe` (10-nov 01:53) compila y arranca |
| GGUF generado | `D:\GEMMA3\sd15-inpaint-ar181-q8_0.gguf` — 3,01 GiB, 1145 tensores |
| **Estado** | **ROTO**: el camino de conversión con override aborta; el GGUF existente está **corrupto** (contenido F32 dentro de bloques Q8_0) |

---

## 2. Lo que se ha establecido (hechos medidos, no supuestos)

### 2.1 El "Q8" no es Q8

Inventario del GGUF generado (`sd_gguf_types.py`, lee solo el header):

| rol | tipo | n | MiB | % del fichero |
|---|---|---|---|---|
| **conv 3x3 / transposed (4D)** | **F32** | **98** | **2247,0** | **73,0 %** |
| ffn (`Linear` 2D) | Q8_0 | 56 | 208,0 | 6,8 % |
| embedding CLIP | F32 | 1 | 144,8 | 4,7 % |
| attn (`Linear` 2D) | Q8_0 | 178 | 123,4 | 4,0 % |
| conv 1x1 (`proj_in/out`) | F32 | 130 | 26,2 | 0,9 % |
| time_embed | F32 | 2 | 7,8 | 0,3 % |
| bias / .scale / LN | F32 | 516 | 1,9 | 0,1 % |

Total 3,01 GiB frente a 4,27 GB del safetensors: **solo −24 %**.
Los tensores dominantes son `(3,3,1280,1280)` ×20 (1125 MiB) y `(3,3,2560,1280)` ×5 (562 MiB).

### 2.2 El fichero está corrupto (bug de upstream, confirmado byte a byte)

`convert.cpp:load_tensor_for_export` crea el tensor destino **ya en el tipo final**
(`ggml_new_tensor(ggml_ctx, job.info.type, ...)`) y luego llama
`ModelLoader::load_tensor`, que **copia bytes crudos sin convertir** (`model_loader.cpp:1448`:
solo valida nombre/offset/forma y asigna el puntero). Resultado: un bloque Q8_0
(34 bytes/bloque) recibe los bytes F32 del fichero. La cabecera dice Q8_0, el contenido
está mal. **Cualquier GGUF creado con `sd-cli -M convert --type q8_0` en este HEAD
está corrupto**, no solo el nuestro.

### 2.3 El modelo se instancia con la arquitectura equivocada

El checkpoint SD1.5 usa **DecisionBlock** (pre-norm + `proj_in` conv 1x1). sd.cpp
instancia `SpatialTransformer` con `use_linear_projection=true` (default SD1 en
`UNetConfig`), así que lee `proj_in` como `Linear` 2D y el exportador nunca ve la conv.

Evidencia falsable: existe `input_blocks.4.1.norm.weight` (hermano que el
Transformer2DModel real no tiene) y `proj_in.weight` es `[640, 640, 1, 1]`.

### 2.4 Qué kernel ejecuta cada cosa en sm_61

- attn/ffn son `Linear` 2D → **MMQ con DP4A** (`mmq-config-pascal-dp4a.cuh` ya existe
  upstream y `mmq.cu:412-424` selecciona por `compiled_arch`; sm_61 tiene DP4A).
- MMVQ solo para `ne11 <= 8` → en la práctica solo `time_embed`.
- **P3 (conv2d implicit-GEMM) sí llega a ejecutarse con este modelo**: el kernel exige
  `kernel->type == GGML_TYPE_F16` (`conv2d.cu:398`) y el loader convierte las convs F32
  a F16 al cargar. Contradice mi informe R3 anterior; **falta confirmarlo con log `-v`**.
- Sin parche, la producción **no** corre el kernel naive: `sd_probe_conv_direct`
  (`conv_direct.cpp:86`) apaga la ruta directa en sm_61 (`cuda_compiled_arch >= 750`,
  `ggml_extend_backend.cpp:594`) → se usa `ggml_conv_2d` = im2col + matmul.

---

## 3. Dónde está el parche (y qué funciona de él)

### Funciona
- **`unet.hpp`**: detección de DecisionBlock por número de ejes `ne[i] != 1`
  (no por `n_dims`, que viene falseado por el padding de `ne[]`). Compila.
- **`convert.cpp` `load_tensor_for_export`**: carga en el tipo origen y llama
  `convert_tensor()` para pasar al tipo destino. Es el fix del §2.2.
- **`convert.cpp` `export_loaded_model`**: aplica `set_wtype_override` antes de
  recolectar (antes las reglas de tipo nunca llegaban al exportador).
- **Diagnóstico**: el reparto de tipos que pide el exportador ya se puede leer
  (`conv3x3 f32 x22`, `linear q8_0 x208`, `proj_in f32 x16`). Ese log es lo que
  demostró que el override no se aplicaba.

### Roto
- **El camino en memoria (`convert_model_in_memory`) aborta** en cuanto se aplica el
  override de tipo. `sd-cli` sale con `0xC0000409` sin ningún log posterior a
  `ggml_cuda_init`. Cuando desactivé el override, el modo memoria funcionaba
  (exportaba 3,01 GiB en ~30 s).
- Hipótesis sin confirmar, por orden:
  1. `load_model_into_memory` (`convert.cpp:475`) dimensiona cada tensor con
     `load_type`; con override mi regla fuerza F32 para los cuantizados, así que el
     buffer pedido es mayor que el que `ModelManager::prepare_params` espera.
  2. `ggml_init` del contexto usa `mem_size = n_tensores * ggml_tensor_overhead()`
     (`convert.cpp:580`); si el override cambia el número de metadatos, se queda corto.
  3. `convert_tensor` con `dst_type` cuantizado exige `n_per_row % 32 == 0`; si alguna
     conv llega con `ne[0]` no múltiplo, `ggml_quantize_chunk` dispara `GGML_ASSERT`.

### Parche incompleto a propósito
- `model_loader.cpp` tiene un `bool exporting_` para no forzar cargas Q8 en inferencia;
  es correcto pero **solo se activa en el camino de exportación por streaming**, que es
  justo el que produce ficheros corruptos.
- Quedan `LOG_INFO("[P61-DEBUG] ...")` de diagnóstico y un contador estático sin usar
  (`p61_dbg_proj`). **No está listo para commit.**

---

## 4. Decisiones que necesito del senior

1. **¿Arreglamos el conversor de sd.cpp o lo rodeamos?** La alternativa es no tocar
   C++ y cuantizar por fuera: cargar el safetensors con `gguf-py`/numpy y escribir el
   GGUF con los tipos correctos. Es más lento pero no depende del loader del proyecto.
2. **¿Las convoluciones deben cuantizarse?** Si sí, hace falta P2/P6 (conv cuantizada
   nativa) porque el kernel implicit-GEMM exige F16. Si no, el "modelo Q8" se queda en
   73 % F32 y la VRAM liberada es marginal.
3. **¿El objetivo es velocidad o VRAM?** Con el 73 % en F32 y las convs pasando a F16
   al cargar, la VRAM apenas baja; la velocidad sí mejora por MMQ en attn/ffn.

---

## 5. Cómo reproducir el estado actual

```bat
REM compilar (borrar el obj de sd.cpp si se toca unet.hpp: lo incluye solo él)
del /q D:\sd-cpp\build-p61-baseline\CMakeFiles\stable-diffusion.dir\src\model\integrations\sd.cpp.obj
D:\vid3d\build-sd-p61.bat build

REM convertir (C: tiene ~0,7 GB libres: usar D:)
D:\sd-cpp\build-p61-baseline\bin\sd-cli.exe -M convert ^
  -m "H:\PTY\AbsoluteReality_1.8.1_INPAINTING.inpainting.safetensors" ^
  -o D:\GEMMA3\out.gguf --type q8_0

REM inventario de tipos
D:\SD-UI\venv\Scripts\python.exe D:\llama-pascal61\wt-fix\pascal61\scripts\sd_gguf_types.py D:\GEMMA3\out.gguf
```

Scripts: `pascal61\scripts\sd_convert_q8.bat`, `pascal61\scripts\sd_gguf_types.py`.

---

## 6. Trampas de entorno (costaron tiempo)

1. **ninja no recompila al tocar `unet.hpp`**: solo lo incluye
   `src/model/integrations/sd.cpp` y ese obj quedaba a fecha vieja. Borrarlo a mano.
2. **`C:` tiene 0,7 GB libres** y el conversor **preasigna** el tamaño de salida
   (`failed to preallocate output file`). Convertir siempre a `D:`.
3. **El GGUF dice Q8_0 pero contiene F32** (§2.2). No medir con él: cualquier número
   de calidad o velocidad saldría de un modelo corrupto.
4. `mt.exe` interpretado dentro de un `echo` hace que `build-sd-p61.bat` devuelva 255
   con el build correcto.
5. CUDA 11.8 + MSVC 2022 aborta; solo VS2019 BuildTools 14.29 funciona.

---

## 7. [R6 — 2026-11-10] Correcciones a este mismo documento y a la R3/R4

Antes de reiniciar el trabajo limpio, tres verificaciones más leídas en código:

### 7.1 El kernel implicit-GEMM NO tiene bug de layout

El documento de la R3/R4 (D2) afirma que `conv2d_implicit_gemm_f16` es NCHW y sd.cpp
pasa WHCN, y que por eso el parche re-interpreta pesos (SSIM esperado 0,6-0,8).
**Leído el kernel, es falso**:

- Peso: `weight[(m0 + row) * k_total + k0 + col]` con `co = m0 + simt_m + i` —
  indexado OC-major, idéntico a `whcn_layout::kernel_index`.
- Input: `offset = (n * ic + ci) * ih * iw + ky * dy * iw + kx * dx`, leído como
  `input[offset + pos0]` — WHCN.
- Output: `output[(blockIdx.z * oc + co) * ow * oh + pos]` — WHCN.

La única inconsistencia real es **escalar**: `pos0 = y0 * sy * iw + x0 * sx` aplica el
stride *antes* de sumar `ky * dy * iw`, así que para `stride > 1` o `dilation > 1` el
índice de input sale en la posición equivocada. El naive usa `whcn_layout` bien.
La UNet de SD1.5 es todo stride 1 / dilation 1, así que **no debería afectar a la
generación**, pero conviene fijarlo antes de medir y probarlo con un caso stride=2.

Consecuencia: **no hay que escribir un kernel WHCN ni un permute previo.** El
"medio día" que pedía la R3 no hace falta para corregir layout; solo para el caso
stride/dilation.

### 7.2 La cuantización de convs es un cambio de una línea, no P2/P6

`Conv2d::init_params` (`ggml_block.hpp:519`) hace literalmente:

```cpp
enum ggml_type wtype = GGML_TYPE_F16;
params["weight"] = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
```

**Ignora `expected_type` por completo.** Esa es la razón real del 73 % F32 — ni el
exportador ni `tensor_should_be_converted` tienen la culpa. Cambiarlo a
`get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F16)` con una guarda de
divisibilidad (`in_channels % ggml_blck_size(wtype) == 0`) basta para que las convs
3x3 vayan a Q8_0.

Aviso: si las convs pasan a Q8_0, `conv2d.cu:398` deja de cumplirse y **P3 deja de
entrar**. La conv cuantizada nativa (P2) sí sería necesaria entonces.

### 7.3 El micro-banco mide una configuración que no se usa

`tests/conv2d_bench.cpp:45` crea el kernel en **F16** y `:32` usa
`ggml_conv_2d_direct`. Con el modelo Q8 las convs llegan F32 (hoy) o Q8_0 (si se
aplica 7.2). El 11-30x del micro-banco no es extrapolable a ninguna de las dos.

### 7.4 Qué se descarta para el da capo

Se reinicia el trabajo con alcance estricto: **sd.cpp + modelo Q8**. No se arrastran
del análisis anterior: MMVQ Pascal (casi no se ejecuta en difusión), imatrix para
`GGML_OP_CONV_2D`, CUDA graphs, ni nada del fork de llama-pascal61 salvo, si acaso,
sus tablas de tuning de sm_61.
