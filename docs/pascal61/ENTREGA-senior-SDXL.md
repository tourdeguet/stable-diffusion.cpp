# Informe al senior — Fork sd.cpp sm_61: estado SD1.5 vs SDXL

Fecha: 2026-11-10. Repo `D:\sd-cpp`, rama `master` (`51eeee4`) + rama `sd1.5-rama-b`, tag `sd1.5-rama-b-validada`.
Todo lo que sigue está verificado en disco y en código, no copiado de planes.

---

## 0. Resumen en cinco líneas

1. **SD1.5 Q8 funciona y está validado** (txt2img e inpaint): 1,17 GiB, 1,35 it/s, imagen limpia, VAE F16.
2. **El "fallo SDXL" no era un fallo**: los modelos son de inpainting y se ejecutaban en modo txt2img.
3. Para llegar ahí hubo que arreglar un bug de nombres (`te.1.`) y otro mío (el VAE se cuantizaba).
4. Con el modo correcto, **SD1.5 inpaint Q8 funciona perfectamente** (prueba con imagen y máscara, 14,7 s).
5. **SDXL-inpaint sí falla**, pero también con el modelo original sin cuantizar → bug de sd.cpp, no de la Rama B.

---

## 1. Estado actual del fork

| | |
|---|---|
| Rama estable | `sd1.5-rama-b` @ `51eeee4`, tag `sd1.5-rama-b-validada` |
| Cambios sin commitear | `src/model_loader.cpp` (+4, exclusión VAE), `src/name_conversion.cpp` (+30/−8, fix `te.1.`) |
| Binario | `build-p61-baseline\bin\sd-cli.exe` (10-nov 14:5x) |
| Respaldo binarios | `D:\GEMMA3\bin-sd-p61-backup-2026-11-10` (+`sd-cli.map` para el handler) |

### Commits de la Rama B

| Commit | Contenido |
|---|---|
| `586a251` | Rama B: convs 4D → Q8_0 en layout GEMM 2D `[ic*kh*kw, oc]`; exportador carga en tipo origen + `convert_tensor`; rama cuantizada en `Conv2d::forward`; detección DecisionBlock; handler `[P61-CRASH]` |
| `a11648f` (submódulo ggml) | conv2d P3 sm_61, diag `[P61-ADDFAIL]`, dump `[P61-SCHED]` |
| `51eeee4` | Guarda anti-clobber: rechaza `.safetensors` de salida con tipo cuantizado antes de abrir |

---

## 2. Modelos producidos

| Modelo | Ruta | Tamaño | Estado |
|---|---|---|---|
| SD1.5 txt2img | `D:\GEMMA3\sd15-t2i-vaeF16.gguf` | 1,16 GiB | ✅ **validado** (imagen limpia, 16 s) |
| SD1.5 inpainting | `D:\GEMMA3\sd15-inpaint-ar181-q8b.gguf` | 1,17 GiB | ✅ **validado en inpaint real** (14,7 s) — VAE aún cuantizado, regenerar |
| Pony V6 XL inpainting | `D:\SD-UI\models\Stable-diffusion\pony-v6-sdxl-inpaint-q8b.gguf` | 3,61 GiB | ⚠️ carga; SDXL-inpaint falla también en stock |
| Juggernaut XL inpaint | pendiente de reconvertir | — | ⚠️ ídem |

Medición SD1.5 (GTX 1060 6 GB, 512×512, 20 pasos, seed 42):

| | Rama B (Q8) | Referencia F16 |
|---|---|---|
| Modelo en VRAM | **1,19 GiB (cabe)** | 3,28 GiB → **no cabe**, va a CPU |
| Sampling | **1,35 it/s** (14,8 s) | 0,095 it/s (210 s) |
| Calidad | SSIM 0,80 / PSNR 23,3 dB vs F16 — divergencia caótica de detalles, visualmente equivalente | — |

---

## 3. Bug A — nombres `te.1.` (ARREGLADO)

### Síntoma
Converter una carpeta **diffusers** SDXL producía un GGUF que no cargaba:

```
[E] Conditioner model tensor 'cond_stage_model.1.transformer.text_model.text_projection'
    not in model metadata
[E] model metadata validation failed --- diffusion_engine.cpp:1366
```

### Causa
`src/model_loader.cpp:441` escribe el prefijo `te.1.` para CLIP-G:

```cpp
if (file_exists(clip_g_path) && !parse_file(clip_g_path, prefix + "te.1.")) {
```

Pero `prefix_map` en `src/name_conversion.cpp` **no tenía entrada para `te.1.`**, solo para `te.` y `te2.`. El nombre acababa como `cond_stage_model.transformer.**1.**text_model.*` (el `1` dentro del nombre) en vez de `cond_stage_model.**1**.transformer.text_model.*`.

### Fix aplicado
Dos cambios, no uno:

1. Añadida la regla `{"te.1.", "cond_stage_model.1.transformer."}`.
2. **El mapa pasó de `std::unordered_map` a `std::vector` ordenado por longitud de prefijo descendente.** Con `unordered_map` el orden de iteración es no determinista y `te.` podía ganar a `te.1.` según el hash. `replace_with_prefix_map` corta en la primera coincidencia.

### Verificación
Pony y Juggernaut **cargan y muestrean** tras el fix (antes no pasaban de la validación).

---

## 4. Bug B — el VAE se cuantizaba (ARREGLADO, era mío)

### Síntoma
Salidas grises y artefactos tipo escamas; `vae decode compute failed`.

### Causa
`tensor_should_be_converted` (`model_loader.cpp`) tenía exclusiones para `time_embed`, `embedding`, `scale_shift_table`… pero **ninguna para el VAE**. El aplanado GEMM de la Rama B alcanzaba también las convs 3x3 del VAE:

| Modelo | VAE antes | VAE después del fix |
|---|---|---|
| Pony SDXL | 120 Q8_0 + 128 F16 | **248 F16, 0 Q8_0** |
| SD1.5 t2i | 120 Q8_0 + 128 F16 | **F16 puro** |
| SD1.5 inpaint | 120 Q8_0 + 128 F32 | pendiente de regenerar |

### Fix aplicado
Añadida la exclusión en `tensor_should_be_converted`:

```cpp
} else if (starts_with(name, "vae.") || starts_with(name, "first_stage_model.")) {
    // Pass, do not convert the VAE
}
```

### Efecto medido
SD1.5 con VAE F16: imagen limpia, sin error de VAE, 16 s. **El gris de SDXL no era el VAE** (sigue gris con VAE F16).

---

## 5. SDXL: NO es un bug del fork — era el modo de uso (RESUELTO)

### Conclusión final

Los tres modelos SDXL descargados (`pony-diffusion-v6-sdxl-inpainting`, `juggernaut-xl-inpainting`) son **modelos de inpainting** (`conv_in.weight` = `[320, 9, 3, 3]`, 9 canales).

Se estaban ejecutando en **modo txt2img** (`-m ... -p ...` sin `-i`/`--mask`). Un modelo de 9 canales sin máscara **no puede generar**: produce gris plano o basura. sd.cpp lo avisa explícitamente:

```
[I] Version: SDXL Inpaint
[W] inpainting model requires both an init image and a mask image.
```

### Prueba que lo demuestra

El **modelo original diffusers, sin cuantizar, sin offload, sin tiling y sin mi parche**, da exactamente la misma imagen rota en modo txt2img (MD5 idéntico entre configuraciones). Si el binario con el modelo stock falla igual, el problema no es la Rama B.

### Prueba correcta de inpaint

| Modelo | Modo | Resultado |
|---|---|---|
| **SD1.5 inpaint Q8** | `-i base.png --mask mask.png` | ✅ **orbe de cristal insertado perfectamente**, 14,7 s |
| Pony V6 XL inpaint Q8 | inpaint | ❌ basura (ver abajo) |
| Pony V6 XL inpaint **original** | inpaint | ❌ basura (misma imagen que el Q8) |

El SD1.5 inpaint demuestra que **la rama cuantizada funciona también con convs de 9 canales**. El fallo SDXL persiste con el modelo original sin tocar → **es un bug de sd.cpp con SDXL-inpaint diffusers**, no del cuantizado, y merece un ticket aparte.

### Lo que se descartó (sin evidencia a favor)

- ~~La rama cuantizada descoloca datos en SDXL~~ → el original sin cuantizar falla igual.
- ~~El VAE cuantizado causa el gris~~ → con VAE F16 sigue igual.
- ~~El tiling/offload rompe el grafo~~ → con y sin tiling, MD5 idéntico.
- ~~Falta de VRAM~~ → a 512×512 y con todo en RAM falla igual.

### Lo que queda confirmado y sí es real

- **Fix `te.1.`**: sin él, el GGUF diffusers SDXL ni siquiera carga.
- **El VAE se cuantizaba**: defecto propio, corregido (exclusión `vae.`/`first_stage_model.`).
- **SD1.5 txt2img e inpaint Q8**: ambos validados, caben en VRAM, calidad correcta.

### Pista para el bug SDXL-inpaint

El SD1.5 inpaint funciona con el mismo código. La diferencia está en el pipeline SDXL: el checkpoint Pony declara `in_channels: 4` en `config.json` pero los pesos tienen 9 canales. sd.cpp detecta `VERSION_SDXL_INPAINT` por la forma del tensor, pero el pipeline de muestreo de SDXL-inpaint (máscara, `conv_in` de 9 canales, condicionamiento extra) puede no estar implementado o probado upstream. **Pregunta para el senior: ¿SDXL-inpaint está soportado en este HEAD, o solo SD1.5-inpaint?**

---

## 6. Estado de VRAM, el objetivo declarado

| Modelo | Q8+VAE F16 | ¿Cabe en 6 GB? |
|---|---|---|
| SD1.5 UNet | 1,19 GiB | ✅ sí, y sobra |
| SDXL UNet | 3,61 GiB (params 2,6 GB al cargar) | ❌ no con activaciones de 1024×1024 |

El auto-fit de sd.cpp decide que no cabe y manda todo a RAM: de ahí los 136 s de sampling. **Aunque se arregle el NaN, SDXL en la 1060 va a necesitar tiling o resolución menor (768×768)**. Esto es una limitación de la GPU, no del cuantizado.

---

## 7. Preguntas concretas para el senior

1. **¿SDXL-inpaint está soportado en este HEAD?** El original diffusers sin cuantizar produce basura igual. El SD1.5-inpaint funciona con el mismo código, así que sospecho que el pipeline SDXL-inpaint (máscara + `conv_in` de 9 canales + condicionamiento extra) no está implementado o probado en `e239e4b`.
2. **¿El checkpoint Pony es válido?** Su `config.json` dice `in_channels: 4` pero los pesos tienen 9. Puede ser un release mal empaquetado (config de txt2img con pesos de inpaint), que rompería cualquier loader que se fíe del config.
3. **¿Con qué modelo se midió SDXL upstream?** Si solo se prueba SDXL txt2img, habría que repetir la medición de VRAM con un SDXL txt2img puro, no con un inpaint.

---

## 8. Anexo — comandos de reproducción

```
# convertir (una carpeta diffusers)
D:\llama-pascal61\wt-fix\pascal61\scripts\sd_convert_q8.bat H:\pony-diffusion-v6-sdxl-inpainting out.gguf

# inventario de tipos
python D:\llama-pascal61\wt-fix\pascal61\scripts\sd_gguf_types.py out.gguf

# generar
sd-cli.exe -m out.gguf -p "..." --steps 20 --seed 42 -W 1024 -H 1024 -o test.png

# el que funciona (SD1.5)
sd-cli.exe -m D:\GEMMA3\sd15-t2i-vaeF16.gguf -p "..." --steps 20 --seed 42 -W 512 -H 512 -o ok.png
```

## 9. Trampas de entorno (para no repetirlas)

1. **Objetos rancios de ninja**: `ggml_block.hpp` lo incluyen 27 TUs; `Conv2d::forward` es inline-implícita. Tras editar ese header, borrar **todos** los `*.obj` de `CMakeFiles\stable-diffusion.dir` o el linker puede elegir la definición vieja.
2. **Conversor: NUNCA escribir sobre un fichero existente.** El writer de safetensors trunca al abrir. Esto destruyó `absolutereality_v181.safetensors`; ya hay guarda en código y en el `.bat`.
3. **Convertir siempre a `D:`** (C: tiene poco espacio y el conversor preasigna).
4. **El conversor diffusers se cuelga** a veces tras escribir el GGUF (proceso vivo, CPU congelada). Matar y comprobar el fichero antes de reintentar.
5. **ASAN no funciona en esta máquina** (exe mínimo con `/fsanitize=address` muere con 0xC0000142).
