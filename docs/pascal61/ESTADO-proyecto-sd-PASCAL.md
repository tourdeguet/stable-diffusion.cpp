# ESTADO DEL PROYECTO — Fork sd.cpp para Pascal (sm_61)

**Fecha:** 2026-11-11 (noche)
**Repo:** `D:\sd-cpp`, `master @ e3f9b60`, árbol limpio. **Publicado:** https://github.com/tourdeguet/stable-diffusion.cpp · tag `p61-sm61-v1` · submódulo en https://github.com/tourdeguet/ggml (rama `p61-conv2d-sm61`)
**Ramas:** `master` (todo) · `sd1.5-rama-b` + tag `sd1.5-rama-b-validada` (en `52d1e1c`)
**Ruta de docs:** `D:\llama-pascal61\wt-fix\pascal61\docs\`

---

## 1. Qué es este fork (una línea)

stable-diffusion.cpp (HEAD upstream `e239e4b`) adaptado a GTX 1060 6 GB (sm_61): cuantización Q8_0 de las convoluciones 3x3 para que el modelo entero quepa en VRAM. La GGML es un submódulo de `leejet/ggml` con parche propio (kernel P3 de llama-pascal61).

## 2. Qué funciona (validado, no supuesto)

| Pieza | Estado | Medida |
|---|---|---|
| SD1.5 txt2img Q8 (`D:\GEMMA3\sd15-t2i-vaeF16.gguf`, 1,16 GiB) | ✅ | 1,35 it/s · 20 pasos 512² ≈ 16 s · SSIM 0,80 vs F16 |
| SD1.5 inpaint Q8 (`D:\GEMMA3\sd15-inpaint-ar181-q8b.gguf`, 1,39 GiB, VAE F32) | ✅ | todo en VRAM (RAM 0 MB) · 13,6 s · orbe insertado correcto |
| Referencia F16 SD1.5 (safetensors) | ❌ no cabe | 0,095 it/s — pesos a CPU (thrashing) |
| Kernel implicit-GEMM F16 (P3, submódulo ggml) | ✅ sin bug de layout | micro-banco 11,8–29,9x; único defecto real: stride/dilation > 1 |
| **SDXL inpaint Q8 completo** (`D:\GEMMA3\bisec\jug-q8-full2.gguf` y `pony-q8-full.gguf`, 3,60 GiB) | ✅ **genera correcto** | UNet en VRAM (2612 MB), 20 pasos 512² = 28,7 s sampling / 44,7 s totales |

**La premisa del proyecto quedó demostrada:** F16 no cabe en 6 GB; Q8_0 sí cabe entero en VRAM y corre ~14x más rápido que el F16 con thrashing.

## 3. Commits del fork

| Commit | Contenido |
|---|---|
| `586a251` | Rama B: exportador correcto (carga en tipo origen + convert_tensor), aplanado GEMM de convs 4D `[ic*kh*kw, oc]`, rama cuantizada en `Conv2d::forward` (im2col F32 → mul_mat(w Q8_0) → reshape + permute(2,0,1,3) → bias), detección DecisionBlock, handler `[P61-CRASH]` |
| `a11648f` | Submódulo ggml: conv2d P3 sm_61, diag `[P61-ADDFAIL]`, dump `[P61-SCHED]` |
| `51eeee4` | Guarda anti-clobber: rechaza salida `.safetensors` con tipo cuantizado ANTES de abrir el fichero (incidente real: destruyó el checkpoint AbsReality v181; restaurado desde CivitAI) |
| `52d1e1c` | Fix `te.1.` (CLIP-G diffusers) + `unordered_map`→`vector` ordenado por longitud de prefijo + exclusión `vae.`/`first_stage_model.` de la cuantización |
| `b3e6519` | **Bug 1 cerrado**: I/O multihilo seguro (fstream propio por escritura), verificación de tamaño + N tensores escritos, carga de origen serializada con cuantización en paralelo, `yield()` en vez de `sleep(10ms)`, aviso de tensores todo-cero al cargar. **3,4× más rápido y byte-idéntico a `--threads 1`** |

## 4. Los dos bugs SDXL — **AMBOS CERRADOS**

### Bug 1 — Conversor I/O multihilo: **CERRADO** en `b3e6519`

Síntoma: la conversión desde carpeta **diffusers** abortaba con **0xC0000409** en el tensor **1680/2641** (frontera UNet→VAE). El GGUF quedaba con el header completo pero el cuerpo truncado/a cero: `pygguf` lo lee sin error, el modelo carga y las latentes salen **gris**. Con `--threads 1` funcionaba (190 s).

**Causas reales (medidas, no supuestas):**

| # | Causa | Arreglo |
|---|---|---|
| 1 | Los N workers compartían **un** `std::fstream` y hacían `seekp`+`write` sin mutex → corrupción del streambuf del CRT → `__fastfail` | Cada escritura abre su propio `fstream` (`r+b`) y comprueba `flush()`/`good()` |
| 2 | `preallocate_output_file` extendía el fichero con `seekp(size-1); put(0)`: un tensor que fallase sin comprobarse dejaba **ceros silenciosos** con EXIT 0 | Verificación final: N tensores escritos **y** tamaño exacto, o error |
| 3 | `ModelLoader` **no es thread-safe**: `load_tensor` reconstruye `file_data` y reabre el fichero de origen | Se serializa **solo** la carga del origen; los bytes van a un buffer propio y `convert_tensor` corre en paralelo |
| 4 | `load_tensors` dormía 10 ms por tensor con `log_progress=false` (2641 × 10 ms ≈ 26 s) | `yield()` |

**Medidas** (`jug-f16.gguf` 6,46 GiB → Q8_0 3,60 GiB):

| Config | Tiempo | Resultado |
|---|---|---|
| `--threads 1` | 177 s | OK |
| 6 hilos, antes del fix | — | crash `0xC0000409` / `0xC0000005` |
| **6 hilos, con el fix** | **52 s (3,4×)** | **byte-idéntico a `--threads 1`** |
| diffusers Pony → F16 (2643 tensores, el caso que abortaba en 1680) | 183 s | EXIT 0, **byte-idéntico** a la referencia de 1 hilo |

**Extra:** `[P61-IO]` avisa al cargar si un tensor tiene 64 bytes todo-cero (O(1) por tensor), así un GGUF corrupto nunca vuelve a pasar en silencio.

### Bug 2 — SDXL gris: **CERRADO. NO ERA UN BUG**. El `jug-f16.gguf` estaba CORRUPTO

**Causa raíz (demostrada con datos):**

```
jug-f16.gguf      → 969 de 2641 tensores F16 son TODO CERO
                     (desde unet.up_blocks.2.resnets.1.norm2.weight, offset 5.127.619.584,
                      hasta EOF: UNet final + VAE completo + CLIP-L + CLIP-G)
pony-f16-st.gguf  → 0 tensores a cero
jug-f16-2.gguf    → 0 tensores a cero (regenerado desde el MISMO checkpoint H:\juggernaut-xl-inpainting)
```

Los CLIP-L y CLIP-G del fichero eran ceros → `c_crossattn` = 0 → las cross-attention anulan los residuales → el sampler converge a ε=0 → latente 0 → `VAE(0) ≈ 128` = **gris plano**. Todo lo demás del diagnóstico fue perseguir ese artefacto.

Pruebas que lo confirman:
- `clipL.raw rms=0,00000` y `clipG.raw rms=0,00000` en jug vs `rms=1,20` / `0,40` en pony, con los mismos tokens (`49406,1628,49407,...`).
- Inspcción de tensores `te.*` en el GGUF: `token_embedding`, `position_embedding` y `final_layer_norm` de ambos encoders, todos a cero en jug; sanos en pony.
- **El mismo conversor, mismo binario, misma ruta `--threads 1` y mismo checkpoint regeneran el fichero SIN ceros** → el exportador no tiene el bug.
- Con `jug-f16-2.gguf` (sano) sale la esfera; con `jug-q8-full2.gguf` (Q8 completo, sano) **también** → la cuantización Q8 queda absuelta definitivamente.

**Por qué se corrompió:** no reproducible con el conversor actual. Hipótesis: el fichero de 6,46 GiB se escribió bajo presión de disco/RAM (el equipo tiene 15,7 GB y en esa sesión se hacían varias conversiones a la vez); el escritor escribe a `fstream` con `seekp`+`write` y si el proceso muere a mitad, los tensores ya escritos (los primeros ~78%) quedan bien y **el resto queda a cero porque el fichero se preasigna/sparse**. **Esto era exactamente el Bug 1** — cerrado en `b3e6519`. **Regla operativa: tras CADA conversión, validar con `scripts/sd_gguf_checkzeros.py`** (el binario ya avisa con `[P61-IO]` al cargar).

**Lo que queda descartado definitivamente (con pruebas limpias):**

| Hipótesis | Prueba | Veredicto |
|---|---|---|
| Q8 rompe SDXL | Q8 completo desde F16 sano → CONTENT ok | **descartado** |
| Convs de la Rama B | solo-convs-Q8 desde F16 sano → mismo resultado que F16 | descartado |
| `ff.`/`attn1`/`attn2` | las tres variantes eran el mismo artefacto de ceros | descartado |
| Text encoder cuantizado | `te.=f16` daba el mismo artefacto (el fichero ya tenía `te.*` a cero) | descartado |
| `label_emb` | ya excluido desde upstream (`model_loader.cpp:1680`) | descartado |
| Backend CUDA / flash-attn / CFG / máscara | todo CPU, flash off, CFG=1, máscara RGB con agujero → mismas conclusiones | descartado |
| Bug del pipeline SDXL-inpaint | inpaint correcto con F16 y Q8 sanos | descartado |

### 4.5 Instrumentación `[P61-DUMP]` (REVERTIDA — histórico del método)

Se usó `ModelTraits::dump_emb` + logs en `get_learned_condition` y en `conditioner.hpp::get_learned_condition_common` (rms por encoder). **Revertida con `git checkout --`**; el árbol está limpio.
Si hay que repetir el diagnóstico:

```
# 1) detectar tensores a cero (primer filtro, 30 s)
python -c "import gguf,numpy as np;r=gguf.GGUFReader(PATH);print(sum(1 for t in r.tensors if (a:=t.data.astype(np.float32)).size and not a.any()),'/',len(r.tensors))"
# 2) si hay ceros: reconvertir con --threads 1 y volver a comprobar
```

## 5. Pipeline Q8 (receta de conversión, verificada)

```
sd-cli -M convert -m <safetensors|diffusers-folder|.gguf> -o <out>.gguf --type q8_0
```

`--threads` es seguro desde `b3e6519` (52 s vs 177 s en GGUF→GGUF, byte-idéntico a `--threads 1`).

- Las convs 4D se exportan aplanadas a 2D `[ic*kh*kw, oc]` (mismo orden de bytes, sin transponer; = índice de patch de `ggml_im2col`).
- La divisibilidad por 32 se evalúa sobre la **fila GEMM** (la puerta del 73 % F32 era `ne[0]=3` sobre la forma 4D, `model_loader.cpp:1680`).
- El grafo: `Conv2d::forward` (`ggml_block.hpp:586`) rama cuantizada = im2col **F32** (NO F16: CUDA rechaza `mul_mat` src1 F16 con src0 cuantizado, `ggml-cuda.cu:7669`) + `mul_mat(w Q8_0, patches)` con peso como src0 + `reshape_4d(OC,OW,OH,N)` + `permute(2,0,1,3)` + cont + bias in-place.
- **Exclusiones vigentes** (`tensor_should_be_converted`): `time_embed.`, `label_emb.` (YA excluido, `model_loader.cpp:1680`), `embedding` (sin punto: cubre `add_embedding`, `time_embedding`), `position_ids.`, `vae.`, `first_stage_model.`.
- El VAE NUNCA se cuantiza (rompe calidad conocida).
- Convertir SIEMPRE a `D:` (C: tiene 0,7 GB).
- Si sale `failed to preallocate output file`, es **disco lleno**, no el conversor.

### 5.5 Cuantización mixta: `ff.net` en Q6_K (Fase 2, medida)

El UNet de SDXL reparte sus parámetros así: **`ff.net` (MLP GEGLU) = 1229,5 M = 47 %**; convs ResNet ~24 %; attention ~28 %. Las MLP son las menos sensibles al error de cuantización, así que son las únicas que merece la pena bajar de Q8_0.

```
sd-cli -M convert -m <fuente.gguf> -o <salida.gguf> --type q8_0 --tensor-type-rules "ff\.net\.=q6_K"
```

Ojo: el nombre del tipo es **`q6_K`** (K mayúscula) y el patrón es un **regex ECMAScript** (`ff\.net\.`, no `ff.net.`). Un nombre mal escrito solo produce `[W] ignoring invalid quant override` y el fichero sale igual que sin la regla.

De los 140 pesos de `ff.net`, **130 van a Q6_K**; los 10 con `ne[0]=640` no son divisibles por el bloque de 256 de las K-quants y se quedan en F16 (32,8 M params, 65 MB). `Q6_K` sobre fila 640 daría basura, pero `tensor_should_be_converted` lo rechaza antes.

Medido a 512² (20 pasos) y 1024² (mismo seed) sobre Juggernaut:

| Variante | Fichero | UNet VRAM | 512² / 20 pasos (total) | 1024² (s/paso en sampling) | PSNR vs Q8 |
|---|---|---|---|---|---|
| `--type q8_0` | 3,60 GiB | 2612 MB (VAE→RAM) | 44,7 s | 5,55 s (8 pasos: 44,4 s) | — |
| **`ff.net` Q6_K** | **3,36 GiB** | **2366 MB (VAE→VRAM)** | **36,6 s (−18 %)** | **4,84 s (−13 %)** | **38,7 dB** |
| `ff.net` Q4_K (extremo) | 3,07 GiB | 2072 MB | no medido | 4,78 s (8 pasos: 38,2 s) | 34,4 dB |

Con Q6_K, 1024² y 20 pasos completos: **96,7 s de sampling / 148,9 s de pared**.

El Q6_K gana en las tres métricas (tamaño, VRAM y velocidad: mueve menos bytes por `mul_mat`) con diferencia visualmente nula. **Es la variante recomendada.** Q4_K solo si algún día hace falta margen para algo más (batch, ControlNet); no se distribuye porque a 1024² ya no es necesario.

**A 1024² SDXL ya entra con el Q8 puro** (`--vae-tiling --backend te=cpu`): la premisa de la Fase 2 ("no cabe") era falsa. La Fase 2 pasó a ser una optimización de velocidad, y eso sí lo consiguió.

## 6. Modelos en disco

**Los GGUF definitivos viven en `D:\SD-UI\models\Stable-diffusion\`** con la nomenclatura `<arquitectura>-<modelo>-<tarea>-<cuantización>.gguf`, validados con `scripts/sd_gguf_checkzeros.py` **y** con inferencia real. Para los SDXL hay dos variantes: `-q8_0` (referencia de máxima calidad) y `-q8_0-ffq6k` (**recomendada**: convoluciones y attention en Q8_0, solo `ff.net` en Q6_K — ver §5.5):

| Modelo | Ruta | Estado |
|---|---|---|
| SD1.5 AbsoluteReality txt2img Q8 | `sd15-absolutereality-t2i-q8_0.gguf` (1,16 GiB, 1131 tensores) | ✅ 0 ceros · 16,3 s / 20 pasos · todo en VRAM (1193 MB) |
| SD1.5 AbsoluteReality inpaint Q8 | `sd15-absolutereality-inpaint-q8_0.gguf` (1,39 GiB, 1145 tensores, VAE F32) | ✅ 0 ceros · 14,2 s / 20 pasos · todo en VRAM (1429 MB) |
| SDXL Juggernaut inpaint Q8 | `sdxl-juggernaut-inpaint-q8_0.gguf` (3,60 GiB, 2641 tensores) | ✅ 0 ceros · UNet 2612 MB en VRAM · 44,7 s / 20 pasos a 512² |
| **SDXL Juggernaut inpaint Q8+ffQ6_K** | `sdxl-juggernaut-inpaint-q8_0-ffq6k.gguf` (3,36 GiB, 2641 tensores) | ✅ **recomendado** · UNet 2366 MB + VAE en VRAM · **36,6 s / 20 pasos a 512²**, 148,9 s a 1024² |
| SDXL Pony v6 inpaint Q8 | `sdxl-pony-v6-inpaint-q8_0.gguf` (3,60 GiB, 2643 tensores) | ✅ 0 ceros · el checkpoint base da textura rara también en F16 |
| **SDXL Pony v6 inpaint Q8+ffQ6_K** | `sdxl-pony-v6-inpaint-q8_0-ffq6k.gguf` (3,36 GiB, 2643 tensores) | ✅ 0 ceros · misma textura que el Q8 (es el checkpoint) |

Catálogo definitivo: **9,75 GiB** en total; cada modelo entra entero en la 1060 (los SD1.5 con todo en VRAM, los SDXL con el UNet en VRAM y CLIP/VAE en RAM).

**Fuentes (no borrar: son las que permiten reconvertir)**

| Modelo | Ruta | Estado |
|---|---|---|
| Juggernaut XL F16 sano | `D:\GEMMA3\jug-f16.gguf` y `D:\GEMMA3\bisec\jug-f16-2.gguf` (idénticos, 6,46 GiB) | ✅ 0 a cero |
| Pony XL F16 sano | `D:\GEMMA3\bisec\pony-f16-st.gguf` y `pony-f16-mt.gguf` (idénticos, 6,46 GiB) | ✅ 0 a cero |
| SD1.5 inpaint origen | `H:\PTY\AbsoluteReality_1.8.1_INPAINTING.inpainting.safetensors` (4,27 GB) | intacto |
| SD1.5 txt2img origen | `D:\SDMAGIC\models\absolutereality_v181.safetensors` (2,13 GB, restaurado de CivitAI) | intacto |
| SDXL diffusers origen | `H:\pony-diffusion-v6-sdxl-inpainting`, `H:\juggernaut-xl-inpainting` | intactos |

**Evidencia de corrupción conservada (borrable):** `D:\GEMMA3\corrupt-backup\pony-v6-sdxl-inpaint-q8b.gguf.CORRUPT` — 968 de 2643 tensores a cero (36,6%), mismo patrón exacto que el `jug-f16.gguf` original.

**Respaldos:** `D:\GEMMA3\bin-sd-p61-backup-2026-11-10` (sd-cli.exe 110 MB, sd-server.exe, conv2d_bench.exe, sd-cli.map 29 MB) · `D:\GEMMA3\sd15-inpaint-q8b-VIEJO.gguf`.

## 7. Trampas de entorno (medidas, para no repetirlas)

1. **Objetos rancios:** `ggml_block.hpp` lo incluyen **27 TUs**; funciones inline-implícitas (Conv2d::forward) quedan instanciadas en cada uno. Tras editarlo, borrar TODOS los `*.obj` de `CMakeFiles\stable-diffusion.dir` y relink, o el linker mezcla versiones F16 rancias con F32 frescas (nos costó toda una mañana de "crashes imposibles").
2. **`/MAP` compartido:** sd-cli y sd-server comparten fichero de map → el segundo link falla en rebuilds incrementales. El map de sd-cli basta para simbolizar.
3. **Handler `[P61-CRASH]`** en `main.cpp`: imprime RIP/RSP + stack simbolizado contra `sd-cli.map`. Conservado (nos salvó la sesión). ASAN no funciona en esta máquina (0xC0000142 en exe mínimo).
4. **C: solo 0,7 GB** — convertir siempre a D:. El F16 intermedio de SDXL ocupa 6,46 GiB.
5. **RAM 15,7 GB** — el modo memoria del conversor (`convert_model_in_memory`) va muy justo; usar streaming.
6. **El conversor diffusers a veces se cuelga** al terminar (destructores estáticos/threads sin join); matar y verificar el fichero.
7. PowerShell 5.1 lee `.ps1` sin BOM como ANSI → solo ASCII en scripts.
8. El micro-banco `conv2d_bench` no se re-enlaza si solo recompilas ggml (liga estática).
9. `st.load_file` del checkpoint entero desde H: se cuelga; usar `safe_open` + get_tensor por prefijo.
10. BLOQUEO TRANSITORIO (11-nov tarde): el agente no recibía salidas de shell/read/grep; notado en sesión #7; reintentar con calma, no en bucle.
11. **`ninja` necesita el entorno MSVC:** lanzarlo directo desde el shell del agente falla con `fatal error C1083: no se puede abrir 'cstdint'/'stdbool.h'`. Usar `cmd /c "call \"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && cd /d D:\sd-cpp\build-p61-baseline && ninja sd-cli"`. **OJO:** no borrar el directorio entero `CMakeFiles\stable-diffusion.dir` (rompe el estado de ninja: "missing and no known rule"); borrar solo los `*.obj` de dentro.
12. **`--flash-attn` solo acepta `on|off|auto`** (no `true|false`); `--backend te=cpu` es alias de `--backend clip=cpu`.
13. **El log de `sd-cli` va a stderr:** redirigir con `2>&1` o se pierde (los `LOG_INFO` de diagnóstico incluidos).
14. **`sd_image_to_tensor` exige `mask` en RGB:** una máscara `L` (escala de grises) da 1 canal y puede reventar; guardar el PNG en RGB.
15. **Los GGUF de SDXL diffusers traen 2 tensores extra** (`te.1.text_model.embeddings.position_ids`, `te.text_model.embeddings.position_ids`): 2643 vs 2641 del conversor GGUF→GGUF. Son inofensivos pero explican el conteo distinto.
17. **El build NO rastrea cambios de headers (CMake 4.2.0-rc3 + ninja en este repo):** `deps = msvc` está declarado pero no se genera ningún `.d`, así que `ninja` responde `no work to do` aunque cambies un `.hpp` incluido. **Un binario viejo con un header nuevo produce crashes imposibles (`0xC0000005` con stack de basura) y funcionalidad que "no aparece" aunque el código está ahí.** Consecuencia práctica: cualquier cambio en `conditioner.hpp`, `unet.hpp`, `ggml_block.hpp` exige `ninja -t clean` (o borrar los `.obj` afectados a mano) antes de reconstruir. La compilación **completa** tarda ~20 min (incluye ggml-cuda); tenlo en cuenta.
18. **Snippet de validación de GGUF (usar tras CADA conversión):**
    ```
    python -c "import gguf,numpy as np;r=gguf.GGUFReader(PATH);z=[t.name for t in r.tensors if (a:=t.data.astype(np.float32)).size and not a.any()];print(len(r.tensors),'tensores;',len(z),'a cero');print(z[:5])"
    ```
    Si hay tensores a cero, el fichero está corrupto: borrar y reconvertir. Un GGUF con el 20% final a cero **carga sin error** y produce gris plano. El binario desde `b3e6519` avisa solo (`[P61-IO] N tensor(es) con cabecera todo-cero`).
20. **`--threads N` en el conversor ya es seguro (desde `b3e6519`)** y 3,4× más rápido en GGUF→GGUF. La verificación de integridad es la red de seguridad real.
21. **Espacio en D:** cada F16 de SDXL ocupa 6,46 GiB y cada Q8 3,60 GiB. Si sale `failed to preallocate output file`, es disco, no el conversor.
19. **Una máscara BLANCA TOTAL no es un test válido:** con inpaint degenerado el modelo converge a latente cero y sale gris 128 aunque todo esté bien. Usar una máscara con agujero (p. ej. cuadrado blanco central sobre negro, RGB).

## 8. Próximos pasos (en orden)

1. ✅ **Bug 2 cerrado** — no era un bug: `jug-f16.gguf` estaba corrupto (969 tensores a cero). SDXL inpaint Q8 completo validado (28,7 s / 20 pasos a 512², UNet en VRAM).
2. ✅ **Bug 1 cerrado** (`b3e6519`): I/O multihilo seguro + verificación de integridad + 3,4× de mejora, byte-idéntico a `--threads 1`.
3. ✅ **GGUF definitivos regenerados y sustituidos** (11-nov noche): `sdxl-juggernaut-inpaint-q8_0.gguf`, `sdxl-juggernaut-inpaint-q8_0-ffq6k.gguf`, `sdxl-pony-v6-inpaint-q8_0.gguf` y `sdxl-pony-v6-inpaint-q8_0-ffq6k.gguf` en `D:\SD-UI\models\Stable-diffusion\`, todos 0 ceros y validados por inferencia. El viejo `pony-v6-sdxl-inpaint-q8b.gguf` resultó tener **968 tensores a cero (36,6%)** — el mismo patrón exacto que `jug-f16.gguf`; movido a `D:\GEMMA3\corrupt-backup\` como evidencia.
3b. ✅ **Fase 2 hecha y medida** (§5.5): SDXL ya entraba a 1024² con Q8; la cuantización mixta `ff.net`=Q6_K baja el fichero a 3,36 GiB, libera 247 MB de UNet, sube el VAE a VRAM y recorta un 18 % el tiempo a 512² con 38,7 dB de PSNR. Es la variante recomendada.
4. **Inpaint de Pony**: su checkpoint base produce salida rara también en F16 (textura de ruido en la región de la máscara; Pony V6 exige *score tags*). Es del checkpoint, no del fork; documentado y cerrado.
5. **VAE del SD1.5 inpaint sale F32** (248 tensores): el autor lo guardó en F32 y la exclusión respeta el tipo origen. Decisión: **no tocarlo** (0,16 GiB, sin coste real).
6. Opcional: bajar más el UNet (convs también a Q6_K, o Q5_K en attention) si algún día hace falta margen para batch > 1 o ControlNet. Medido: `ff.net` Q4_K deja el UNet en 2072 MB a 34,4 dB de PSNR.
7. ✅ Instrumentación `[P61-DUMP]` revertida; árbol limpio en `b3e6519`.

## 9. Documentos relacionados

- `PLAN-sd-CONVS-Q8-RAMA-B.md` — plan y RESULTADO de la Rama B (sección §8)
- `ENTREGA-senior-CONVERSION-Q8.md` — historial de la conversión Q8 (R6)
- `ENTREGA-senior-SDXL.md` — informe SDXL (R1; **superado en su diagnóstico final**: el gris no era bug de sd.cpp ni del pipeline, era un GGUF corrupto. Ver §4 de este documento).
- `P3-CONV2D-SM61.md` — kernel implicit-GEMM (resultado del micro-banco)
- `PLAN-INPAINTING-AR181.md` — plan v4 original (dos suposiciones corregidas desde entonces)

---

## 10. Lecciones de la sesión #8 (para no repetirlas)

- **Validar el GGUF antes de diagnosticar.** El "gris de SDXL" que ocupó dos sesiones era un checkpoint con 969 tensores a cero (el 36,7% del fichero). Un chequeo de 30 s (`sd_gguf_checkzeros` / snippet §7.18) lo habría cazado en el primer minuto.
- **Un fichero truncado/corrupto carga sin error.** El cargador lee el header, ve los 2641 tensores y sigue; los ceros no provocan ningún aviso.
- **El build no son fiables los incrementales con headers** (ver §7.17): si un cambio de `.hpp` "no tiene efecto", es este el motivo, no el código.
- **Controles antes que hipótesis:** SD1.5 con el mismo runner y Pony con el mismo binario demostraron que el entorno estaba bien; sin esos dos controles el trabajo se habría ido por el camino del backend o de la cuantización.
- **Un fix "no funciona" puede ser un fix en el sitio equivocado.** El primer parche serializó la función entera (170 s, sin mejora visible: parecía que el multihilo no servía). Al estrechar el mutex volvió el crash, y eso localizó al culpable real (`ModelLoader` + arena ggml compartida, no `convert_tensor`). Comparar tiempos sin entender *qué* se serializó casi lleva a descartar el arreglo correcto.
- **La prueba de corrección de un conversor es byte-identidad contra la ruta lenta**, no "termina sin crash".

---

## 11. Publicación y entorno de pruebas (2026-11-11 noche)

### 11.1 Estado publicado

| Elemento | Valor |
|---|---|
| Repo | `https://github.com/tourdeguet/stable-diffusion.cpp` |
| Tag | `p61-sm61-v1` → `e3f9b60` |
| Submódulo `ggml` | `https://github.com/tourdeguet/ggml`, rama `p61-conv2d-sm61` (commit `a11648f`) |
| Base | upstream `e239e4b` + 8 commits (59 commits por detrás de upstream, a propósito) |
| Licencia | MIT (heredada) |

**Tres bloqueantes que había que cerrar antes de publicar, y cómo se cerraron:**

1. **Submódulo irresoluble.** El gitlink apuntaba a `a11648f`, un commit que no existe en `leejet/ggml`: `git clone --recursive` moría con *reference is not a tree*. Ahora `.gitmodules` apunta al fork propio con ese commit en una rama. **Verificado con un clon real desde GitHub.**
2. **Sin copia remota.** Los commits solo existían en un disco.
3. **Docs fuera del repo.** Ahora `PASCAL61.md` en la raíz + `docs/pascal61/` + `scripts/sd_gguf_checkzeros.py`.

**Build limpio verificado desde el clon** (`D:\p61-clone-test`): `cmake` + `ninja` + `nvcc` completan `sd-cli` (434 targets) y `sd-server` (19 targets) sin errores. Trampa descubierta al hacerlo: `ggml` tiene a su vez el submódulo `src/ggml-cuda/vendors/cutlass`; con un clon de un solo nivel falla con *CUTLASS headers not found*. Hay que clonar `--recursive` en todos los niveles.

### 11.2 Entorno de pruebas con interfaz

`D:\SD-UI` es **Forge 1.10.1**, que tiene loader GGUF propio (`backend/operations_gguf.py`, soporta Q8_0 y Q6_K). **Aun así no puede cargar estos ficheros**, por dos motivos independientes:

1. Las convs 3x3 van aplanadas a layout GEMM: `unet.down_blocks.0.resnets.0.conv1.weight` tiene forma `(2880, 320)`; Forge espera `[320, 320, 3, 3]`.
2. Nomenclatura sd.cpp (`unet.*`, `vae.*`, `te.*`) frente al esquema A1111 (`model.diffusion_model.*`, `first_stage_model.*`).

Y aunque se pudiera, sería contraproducente: Forge es PyTorch, descomprimiría el Q8 por operación en fp16, sin kernel P3 y sin el reparto de memoria de sd.cpp. En 6 GB no cabría.

**Solución adoptada:** probar con el `sd-server` del fork, que sí entiende estos GGUF y además expone `/sdapi/v1/...` (compatible A1111), `/v1/images/generations` (OpenAI) y `/sdcpp/v1/...` (nativa), con frontend web embebido.

Instalación creada:

```
D:\SD-UI\p61\bin\sd-server.exe   111 MB (commit e3f9b60, compilado desde el clon)
D:\SD-UI\p61\bin\sd-cli.exe      110 MB
D:\SD-UI\p61\p61-server.bat      lanzador (ASCII puro + CRLF; valida modelo y abre el navegador)
```

Verificado end-to-end: `/sdcpp/v1/capabilities` y `/sdapi/v1/sd-models` responden, `POST /sdapi/v1/txt2img` generó una esfera de cristal correcta en 21,8 s (512², 8 pasos), y `/` sirve el frontend (`stable-diffusion.cpp WebUI`).

Uso: `D:\SD-UI\p61\p61-server.bat [ruta\modelo.gguf]` → http://127.0.0.1:1234
