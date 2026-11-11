# Fork Pascal61 / sm_61 — stable-diffusion.cpp en GTX 1060 de 6 GB

Fork de [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
para GPUs **Pascal (compute capability 6.1)**, en concreto la GTX 1060 de 6 GB.

**Base:** upstream `e239e4b` (2026-11). El fork tiene 5 commits propios; está
deliberadamente anclado a esa base y no se ha rebaseado sobre upstream todavía.
`origin` apunta a upstream para integrar cambios de forma deliberada.

## Qué añade

| Commit | Contenido |
|---|---|
| `586a251` | **Rama B**: exportador correcto (carga en el tipo origen y convierte después), convs 4D aplanadas a layout GEMM `[ic*kh*kw, oc]`, y rama cuantizada en `Conv2d::forward` (im2col F32 → `mul_mat` con peso cuantizado → reshape/permute → bias) |
| `a11648f` | *(submódulo ggml)* kernel **implicit-GEMM conv2d** para sm_61, más diagnóstico `[P61-ADDFAIL]` y volcado `[P61-SCHED]` |
| `51eeee4` | Guarda anti-clobber: rechaza salida `.safetensors` con tipos cuantizados **antes** de abrir el fichero |
| `52d1e1c` | Fix del mapeo de nombres `te.1.` (CLIP-G de diffusers) y exclusión del VAE de la cuantización |
| `5ae5d0d` | **I/O del conversor**: escritura segura multihilo, verificación de integridad, y aviso al cargar si un tensor está todo a cero |
| `3d415d9` | `.gitmodules` apunta al fork de `ggml` que contiene el kernel P3 |

El submódulo `ggml` **no** es el de upstream: apunta a
[`tourdeguet/ggml`](https://github.com/tourdeguet/ggml), rama `p61-conv2d-sm61`.
`git clone --recursive` funciona.

## Por qué existe

Una GTX 1060 tiene 6 GB. Stable Diffusion 1.5 en F16 no sólo no cabe: los pesos
caen a CPU y el resultado es *thrashing* a **0,095 it/s**. Con las convoluciones
3x3 en Q8_0 el modelo entero entra en VRAM y va a **1,35 it/s** (~14x).

Para SDXL el UNet cuantizado son 2,4 GB y también entra, incluyendo **1024²**.

## Resultados medidos (GTX 1060 6 GB, 6 núcleos)

| Modelo | Tamaño | Rendimiento |
|---|---|---|
| SD1.5 txt2img Q8_0 | 1,16 GiB | 16,3 s / 20 pasos a 512², todo en VRAM |
| SD1.5 inpaint Q8_0 | 1,39 GiB | 14,2 s / 20 pasos a 512², todo en VRAM |
| SDXL inpaint Q8_0 | 3,60 GiB | UNet 2612 MB en VRAM, 44,7 s / 20 pasos a 512² |
| SDXL inpaint Q8_0 + `ff.net` Q6_K | 3,36 GiB | UNet 2366 MB, **36,6 s** a 512² · 148,9 s a 1024² |

La cuantización mixta (todo en Q8_0 salvo las MLP `ff.net` en Q6_K) es la
recomendada: 18 % más rápida y con PSNR de 38,7 dB frente al Q8 puro
(visualmente indistinguible).

```bash
# conversión recomendada
sd-cli -M convert -m model.safetensors -o model.gguf --type q8_0

# variante mixta para SDXL (ff.net en Q6_K; el tipo lleva la K mayúscula)
sd-cli -M convert -m model.gguf -o model-mix.gguf --type q8_0 \
       --tensor-type-rules "ff\.net\.=q6_K"
```

## Validar siempre el GGUF

Un GGUF con tensores a cero **carga sin error** y produce latentes planas (gris
128). Eso costó dos sesiones de diagnóstico en este proyecto: el culpable era un
fichero con el 36 % de sus pesos a cero, no el código.

```bash
python scripts/sd_gguf_checkzeros.py model.gguf   # exit 2 = corrupto
```

Desde `5ae5d0d` el binario también avisa al cargar
(`[P61-IO] N tensor(es) con cabecera todo-cero`).

## Documentación

- [`docs/pascal61/ESTADO-proyecto-sd-PASCAL.md`](docs/pascal61/ESTADO-proyecto-sd-PASCAL.md) — estado completo, medidas, trampas de entorno y lecciones
- [`docs/pascal61/P3-CONV2D-SM61.md`](docs/pascal61/P3-CONV2D-SM61.md) — el kernel implicit-GEMM y su micro-banco
- [`docs/pascal61/PLAN-sd-CONVS-Q8-RAMA-B.md`](docs/pascal61/PLAN-sd-CONVS-Q8-RAMA-B.md) — diseño de la rama cuantizada
- [`docs/pascal61/ENTREGA-senior-CONVERSION-Q8.md`](docs/pascal61/ENTREGA-senior-CONVERSION-Q8.md) — historial de la conversión
- [`docs/pascal61/ENTREGA-senior-SDXL.md`](docs/pascal61/ENTREGA-senior-SDXL.md) — informe SDXL (superado: el "gris" era un GGUF corrupto)
- [`docs/pascal61/PLAN-INPAINTING-AR181.md`](docs/pascal61/PLAN-INPAINTING-AR181.md) — plan original de inpainting

## Compilar (Windows, MSVC)

El build usa CUDA; en esta máquina se genera con CMake + ninja y `vcvars64`.
Ojo: en este árbol ninja no rastrea dependencias de cabeceras (`deps = msvc`
declarado sin generar `.d`), así que **tras tocar un `.hpp` hay que limpiar los
`.obj` afectados** o el enlazado mezcla versiones distintas del mismo header y
produce fallos imposibles de depurar.

```bat
REM 1) clonar con TODOS los niveles de submodulos (ggml trae cutlass dentro)
git clone --recursive https://github.com/tourdeguet/stable-diffusion.cpp.git
cd stable-diffusion.cpp
git submodule update --init --recursive --depth 1

REM 2) compilar
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sd-cli -j
```

`ggml` tiene a su vez el submódulo `src/ggml-cuda/vendors/cutlass`: si solo se
inicializa el primer nivel, `cmake` falla con *CUTLASS headers not found at ...
Run git submodule update --init --recursive*.

## Licencia

MIT, la misma que upstream. Ver [`LICENSE`](LICENSE).
