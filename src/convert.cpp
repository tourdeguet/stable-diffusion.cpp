#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_io/streaming_writer.h"
#include "model_loader.h"
#include "model_manager.h"
#include "name_conversion.h"
#include "runtime/imatrix.h"

struct TensorExportInfo {
    TensorStorage storage;
    ggml_type type;
    // [P61] Rama B: cuando storage se aplana a layout GEMM 2D, estos son los
    // ne/n_dims originales (4D del fichero) para poder cargar el tensor.
    bool flattened = false;
    int n_dims_src = 0;
    int64_t ne_src[4] = {1, 1, 1, 1};
};

struct TensorExportJob {
    TensorExportInfo info;
    std::vector<uint8_t> data;
    std::string error;
    bool success = false;
};

static ggml_type get_export_tensor_type(ModelLoader& model_loader,
                                        const TensorStorage& tensor_storage,
                                        ggml_type type,
                                        const TensorTypeRules& tensor_type_rules) {
    const std::string& name = tensor_storage.name;
    ggml_type tensor_type   = tensor_storage.type;
    ggml_type dst_type      = type;

    for (const auto& tensor_type_rule : tensor_type_rules) {
        std::regex pattern(tensor_type_rule.first);
        if (std::regex_search(name, pattern)) {
            dst_type = tensor_type_rule.second;
            break;
        }
    }

    if (model_loader.tensor_should_be_converted(tensor_storage, dst_type)) {
        tensor_type = dst_type;
    }

    return tensor_type;
}

static bool collect_tensors_for_export(ModelLoader& model_loader,
                                       ggml_type type,
                                       const TensorTypeRules& tensor_type_rules,
                                       std::vector<TensorExportInfo>& tensors) {
    tensors.clear();
    tensors.reserve(model_loader.get_tensor_storage_map().size());
    for (const auto& kv : model_loader.get_tensor_storage_map()) {
        const TensorStorage& tensor_storage = kv.second;
        TensorExportInfo info;
        info.storage = tensor_storage;
        info.type    = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);
        // [P61] Rama B (VRAM): pesos 4D de conv con kw==kh en {1,3} se exportan
        // aplastados a layout GEMM 2D [ic*kh*kw, oc] para poder cuantizarlos:
        // Q8_0 exige filas de 32 elementos contiguos y ne[0]=3 no lo cumple.
        if (ggml_is_quantized(info.type) && tensor_storage.n_dims == 4 &&
            tensor_storage.ne[0] == tensor_storage.ne[1] &&
            (tensor_storage.ne[0] == 1 || tensor_storage.ne[0] == 3) &&
            (tensor_storage.ne[0] * tensor_storage.ne[1] * tensor_storage.ne[2]) %
                ggml_blck_size(info.type) ==
                0) {
            LOG_INFO("[p61] conv weight '%s' flattened to GEMM layout [%lld, %lld] for %s",
                     tensor_storage.name.c_str(),
                     (long long int)(tensor_storage.ne[0] * tensor_storage.ne[1] * tensor_storage.ne[2]),
                     (long long int)tensor_storage.ne[3],
                     ggml_type_name(info.type));
            info.storage.n_dims = 2;
            info.storage.ne[0]  = tensor_storage.ne[0] * tensor_storage.ne[1] * tensor_storage.ne[2];
            info.storage.ne[1]  = tensor_storage.ne[3];
            info.storage.ne[2]  = 1;
            info.storage.ne[3]  = 1;
            info.flattened      = true;
            info.n_dims_src     = tensor_storage.n_dims;
            for (int k = 0; k < 4; k++) {
                info.ne_src[k] = tensor_storage.ne[k];
            }
        }
        tensors.push_back(std::move(info));
    }
    LOG_INFO("collected %zu tensors for export", tensors.size());
    return true;
}

static size_t export_tensor_nbytes(const TensorExportInfo& info) {
    TensorStorage output_storage = info.storage;
    output_storage.type          = info.type;
    return static_cast<size_t>(output_storage.nbytes());
}

static TensorWritePlan tensor_write_plan_from_export_info(const TensorExportInfo& info, const std::string& name) {
    TensorWritePlan plan;
    plan.name   = name;
    plan.type   = info.type;
    plan.n_dims = info.storage.n_dims;
    for (int i = 0; i < SD_MAX_DIMS; i++) {
        plan.ne[i] = info.storage.ne[i];
    }
    return plan;
}

static std::vector<TensorWritePlan> tensor_write_plans_from_export_infos(const std::vector<TensorExportInfo>& tensors,
                                                                         bool convert_name,
                                                                         SDVersion version) {
    std::vector<TensorWritePlan> plans;
    plans.reserve(tensors.size());
    for (const TensorExportInfo& info : tensors) {
        const std::string name = convert_name ? convert_tensor_name(info.storage.name, version) : info.storage.name;
        plans.push_back(tensor_write_plan_from_export_info(info, name));
    }
    return plans;
}

static bool preallocate_output_file(const std::string& output_path, uint64_t file_size, std::string* error) {
    if (file_size == 0) {
        return true;
    }

    std::fstream file(std::filesystem::u8path(output_path), std::ios::binary | std::ios::in | std::ios::out);
    if (!file.is_open()) {
        if (error != nullptr) {
            *error = "failed to open output file '" + output_path + "' for preallocation";
        }
        return false;
    }

    // This portable fallback sets the final file size. A platform-specific
    // posix_fallocate/ftruncate path can replace it later.
    file.seekp(static_cast<std::streamoff>(file_size - 1), std::ios::beg);
    file.put('\0');
    file.flush();
    if (!file) {
        if (error != nullptr) {
            *error = "failed to preallocate output file '" + output_path + "'";
        }
        return false;
    }
    return true;
}

static bool load_tensor_for_export(ModelLoader& model_loader, TensorExportJob& job) {
    size_t mem_size = 4 * 1024 * 1024;
    mem_size += ggml_tensor_overhead();
    mem_size += static_cast<size_t>(job.info.storage.nbytes());

    // [P61-IO] Todo lo que toca ModelLoader (load_tensor reconstruye file_data, valida
    // fuentes) y el arena de ggml por tensor se hace bajo este mutex. Los bytes crudos
    // del origen se copian a un buffer propio y la CUANTIZACION (convert_tensor) se
    // ejecuta fuera del lock, en paralelo entre hilos. Serializar la funcion entera
    // daba 170s (igual que --threads 1); paralelizar sin lock daba 0xC0000409.
    static std::mutex src_mutex;
    std::vector<uint8_t> src_bytes;
    ggml_type src_type   = job.info.storage.type;
    std::vector<float> imatrix;

    {
        std::lock_guard<std::mutex> src_lock(src_mutex);

        ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});
        if (ggml_ctx == nullptr) {
            job.error = "ggml_init failed for tensor '" + job.info.storage.name + "'";
            return false;
        }

        // [P61] Cargar en el tipo ORIGEN y convertir a job.info.type. Antes se creaba el tensor
        // ya en el tipo destino (Q8_0) y se le pasaba a load_tensor, que no convierte: los
        // bytes F32 del fichero acababan copiados dentro de un bloque Q8_0 (contenido corrupto
        // aunque la cabecera GGUF declarase Q8_0).
        // [P61] Rama B: el tensor destino siempre se crea con la forma ORIGINAL del fichero
        // (load_tensor valida shape contra current_tensor_storage); el orden de bytes crudo
        // de una conv [oc][ic][kh][kw] ya es el layout GEMM [ic*kh*kw, oc] por verificacion
        // del kernel de im2col (ops.cpp:7014), asi que la conversion usa la fila aplanada.
        const int n_load_dims = job.info.flattened ? job.info.n_dims_src : job.info.storage.n_dims;
        int64_t ne_load[4]    = {1, 1, 1, 1};
        for (int k = 0; k < n_load_dims; k++) {
            ne_load[k] = job.info.flattened ? job.info.ne_src[k] : job.info.storage.ne[k];
        }
        TensorStorage src_storage = job.info.storage;
        src_storage.n_dims        = n_load_dims;
        for (int k = 0; k < 4; k++) {
            src_storage.ne[k] = ne_load[k];
        }

        ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, job.info.storage.type, n_load_dims, ne_load);
        if (tensor == nullptr) {
            ggml_free(ggml_ctx);
            job.error = "ggml_new_tensor failed for tensor '" + job.info.storage.name + "'";
            return false;
        }
        ggml_set_name(tensor, job.info.storage.name.c_str());

        const size_t src_nbytes = ggml_nbytes(tensor);
        if (src_nbytes > 0) {
            if (!model_loader.load_tensor(src_storage, tensor)) {
                ggml_free(ggml_ctx);
                job.error = "failed to load tensor '" + job.info.storage.name + "'";
                return false;
            }
            src_bytes.assign(static_cast<const uint8_t*>(tensor->data),
                             static_cast<const uint8_t*>(tensor->data) + src_nbytes);
        }
        ggml_free(ggml_ctx);

        imatrix = get_imatrix_collector().get_values(
            convert_tensor_name(job.info.storage.name, model_loader.get_sd_version()));
    }

    if (src_bytes.empty()) {
        job.error = "empty source tensor '" + job.info.storage.name + "'";
        return false;
    }

    const size_t dst_nbytes = export_tensor_nbytes(job.info);
    job.data.resize(dst_nbytes);
    int64_t n_per_row = job.info.storage.ne[0];
    int64_t n_rows    = job.info.storage.nelements() / n_per_row;
    convert_tensor(src_bytes.data(),
                   src_type,
                   job.data.data(),
                   job.info.type,
                   (int)n_rows,
                   (int)n_per_row,
                   std::move(imatrix));
    return true;
}

static bool export_tensor_from_memory(const TensorExportInfo& info,
                                      const std::map<std::string, ggml_tensor*>& mem,
                                      SDVersion version,
                                      std::vector<uint8_t>& out) {
    auto found = mem.find(info.storage.name);
    if (found == mem.end() || found->second == nullptr || found->second->data == nullptr) {
        return false;
    }
    const TensorStorage& storage = info.storage;
    out.resize(export_tensor_nbytes(info));
    if (out.size() > 0) {
        std::vector<float> imatrix = get_imatrix_collector().get_values(convert_tensor_name(info.storage.name, version));
        convert_tensor(found->second->data,
                       found->second->type,
                       out.data(),
                       info.type,
                       (int)(storage.nelements() / storage.ne[0]),
                       (int)storage.ne[0],
                       std::move(imatrix));
    }
    return true;
}

static bool stream_tensor_data(ModelLoader& model_loader,
                               const std::string& output_path,
                               const std::vector<TensorExportInfo>& tensors,
                               const StreamingModelWriter& writer,
                               int n_threads,
                               const std::map<std::string, ggml_tensor*>* mem,
                               std::string* error) {
    n_threads = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    n_threads = std::max(1, n_threads);
    LOG_INFO("streaming convert with %d threads", n_threads);

    const SDVersion version = model_loader.get_sd_version();

    int64_t start_time       = ggml_time_ms();
    uint64_t bytes_written   = 0;
    size_t tensors_written   = 0;
    size_t next_tensor_index = 0;
    bool failed              = false;
    std::string failure;

    const size_t memory_budget = 1024ull * 1024ull * 1024ull;
    size_t reserved_bytes      = 0;

    std::mutex work_mutex;
    std::mutex progress_mutex;
    // [P61-IO] Cada escritura abre su propio fichero (no se comparte el streambuf del CRT).
    std::condition_variable memory_cv;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    // [P61-IO] Marcas de escritura: (a) estado de progreso por tensor, (b) verificacion
    // final de que TODOS los tensores se escribieron (un hueco de preasignacion es ceros).
    std::vector<uint8_t> tensor_done(tensors.size(), 0);
    size_t tensors_done_count = 0;

    auto reserve_memory = [&](size_t bytes) -> bool {
        std::unique_lock<std::mutex> lock(work_mutex);
        memory_cv.wait(lock, [&]() {
            return failed || reserved_bytes == 0 || reserved_bytes + bytes <= memory_budget;
        });
        if (failed) {
            return false;
        }
        reserved_bytes += bytes;
        return true;
    };

    auto release_memory = [&](size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            reserved_bytes -= std::min(reserved_bytes, bytes);
        }
        memory_cv.notify_all();
    };

    auto fail = [&](const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            if (!failed) {
                failed  = true;
                failure = message;
            }
        }
        memory_cv.notify_all();
    };

    for (int worker = 0; worker < n_threads; worker++) {
        workers.emplace_back([&]() {
            while (true) {
                size_t tensor_index = 0;
                {
                    std::lock_guard<std::mutex> lock(work_mutex);
                    if (failed || next_tensor_index >= tensors.size()) {
                        return;
                    }
                    tensor_index = next_tensor_index++;
                }

                const size_t tensor_bytes = export_tensor_nbytes(tensors[tensor_index]);
                if (!reserve_memory(tensor_bytes)) {
                    return;
                }

                TensorExportJob job;
                job.info = tensors[tensor_index];
                try {
                    job.success = (mem != nullptr) && export_tensor_from_memory(job.info, *mem, version, job.data);
                    if (!job.success) {
                        job.success = load_tensor_for_export(model_loader, job);
                    }
                } catch (const std::exception& e) {
                    job.error   = e.what();
                    job.success = false;
                }

                if (!job.success) {
                    release_memory(tensor_bytes);
                    fail(job.error.empty() ? "streaming conversion failed" : job.error);
                    return;
                }

                // [P61-IO] Fichero propio por escritura: std::fstream compartido entre
                // hilos corrompe el streambuf del CRT (0xC0000409) y, peor, si la escritura
                // falla sin comprobarse, el hueco preasignado por preallocate_output_file
                // queda a CERO y el GGUF termina con EXIT 0 (incidente jug-f16.gguf).
                std::string write_error;
                bool write_ok = false;
                {
                    std::fstream output_file(std::filesystem::u8path(output_path),
                                             std::ios::binary | std::ios::in | std::ios::out);
                    if (!output_file.is_open()) {
                        write_error = "failed to open output file '" + output_path + "' for tensor writing";
                    } else {
                        write_ok = writer.write_tensor(output_file,
                                                       tensor_index,
                                                       job.data.empty() ? nullptr : job.data.data(),
                                                       job.data.size(),
                                                       &write_error);
                        output_file.flush();
                        if (write_ok && !output_file.good()) {
                            write_ok     = false;
                            write_error = "failed to flush tensor '" + job.info.storage.name +
                                          "' (disco lleno?)";
                        }
                    }
                }
                if (!write_ok) {
                    release_memory(tensor_bytes);
                    fail(write_error.empty() ? "streaming conversion write failed" : write_error);
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(work_mutex);
                    tensor_done[tensor_index] = 1;
                    tensors_done_count++;
                }

                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    bytes_written += job.data.size();
                    tensors_written++;
                    float elapsed_seconds = (ggml_time_ms() - start_time) / 1000.0f;
                    pretty_bytes_progress(static_cast<int>(tensors_written),
                                          static_cast<int>(tensors.size()),
                                          bytes_written,
                                          elapsed_seconds);
                }
                release_memory(tensor_bytes);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    printf("\n");
    if (failed) {
        if (error != nullptr) {
            *error = failure;
        }
        return false;
    }

    // [P61-IO] Verificacion final: los N tensores se escribieron y el fichero tiene el
    // tamano exacto. Sin esto, un hueco de preallocate_output_file (ceros) pasa por bueno.
    if (tensors_done_count != tensors.size()) {
        if (error != nullptr) {
            *error = "streaming conversion incomplete: " + std::to_string(tensors_done_count) +
                     " of " + std::to_string(tensors.size()) + " tensors written";
        }
        return false;
    }
    {
        std::error_code ec;
        const auto expected = static_cast<uintmax_t>(writer.file_size());
        const auto actual   = std::filesystem::file_size(std::filesystem::u8path(output_path), ec);
        if (ec || actual != expected) {
            if (error != nullptr) {
                *error = "output file size mismatch: expected " + std::to_string(expected) +
                         " bytes, got " + (ec ? std::string("<stat failed>") : std::to_string(actual));
            }
            return false;
        }
    }

    LOG_INFO("streaming conversion completed, taking %.2fs", (ggml_time_ms() - start_time) / 1000.f);
    return true;
}

static bool write_model_file_streaming(ModelLoader& model_loader,
                                       const std::string& output_path,
                                       const std::vector<TensorExportInfo>& tensors,
                                       StreamingModelWriter& writer,
                                       int n_threads,
                                       bool convert_name,
                                       const std::map<std::string, ggml_tensor*>* mem,
                                       std::string* error) {
    std::vector<TensorWritePlan> plans = tensor_write_plans_from_export_infos(tensors, convert_name, model_loader.get_sd_version());
    if (!writer.write_metadata(output_path, plans, error)) {
        return false;
    }
    if (!preallocate_output_file(output_path, writer.file_size(), error)) {
        return false;
    }
    model_loader.process_model_files(false, false);
    return stream_tensor_data(model_loader, output_path, tensors, writer, n_threads, mem, error);
}

static bool init_convert_path(ModelLoader& model_loader, const char* path, const char* prefix, bool& loaded_any) {
    if (path == nullptr || strlen(path) == 0) {
        return true;
    }
    if (!model_loader.init_from_file(path, prefix)) {
        LOG_ERROR("init model loader from file failed: '%s'", path);
        return false;
    }
    loaded_any = true;
    return true;
}

static bool export_loaded_model(ModelLoader& model_loader,
                                const char* output_path,
                                sd_type_t output_type,
                                const char* tensor_type_rules,
                                int n_threads,
                                bool convert_name,
                                const std::map<std::string, ggml_tensor*>* mem = nullptr) {
    ggml_type type             = sd_type_to_ggml_type(output_type);
    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    // [P61] Guarda anti-truncamiento: safetensors no soporta tipos cuantizados. Si
    // se pide q8_0/q6/q4 con salida .safetensors, rechazar ANTES de abrir el fichero
    // (el writer trunca al abrir; un fallo a mitad de conversion destruye la
    // salida si coincide con el fichero de origen). Solo F16/F32/BF16 son legales.
    if (output_is_safetensors && type != GGML_TYPE_COUNT && type != GGML_TYPE_F16 &&
        type != GGML_TYPE_F32 && type != GGML_TYPE_BF16) {
        LOG_ERROR("--type %s requires GGUF output: safetensors cannot store it "
                  "(output: '%s')",
                  ggml_type_name(type),
                  output_path);
        return false;
    }

    // [P61] Aplicar el tipo pedido (--type / --tensor-type) al storage antes de recolectar,
    // igual que hace la ruta de inferencia. Sin esto, las reglas de tipo nunca alcanzan
    // a Conv2d::init_params y el fichero sale con las convs intactas en F32.
    if (!output_is_safetensors && (type != GGML_TYPE_COUNT || tensor_type_rules != nullptr)) {
        model_loader.set_exporting(true);
        model_loader.set_wtype_override(type, tensor_type_rules != nullptr ? tensor_type_rules : "");
    }

    std::vector<TensorExportInfo> tensors;
    bool success = collect_tensors_for_export(model_loader, type, type_rules, tensors);

    {
        std::map<std::string, int> stat;
        for (const auto& ti : tensors) {
            const std::string& nm = ti.storage.name;
            std::string g = "otro";
            if (nm.find(".proj_in.weight") != std::string::npos) g = "proj_in";
            else if (nm.find(".proj_out.weight") != std::string::npos) g = "proj_out";
            else if (nm.find("in_layers.2.weight") != std::string::npos) g = "conv3x3";
            else if (nm.find("transformer_blocks") != std::string::npos && nm.find(".weight") != std::string::npos) g = "linear";
            if (g != "otro") stat[g + " " + ggml_type_name(ti.type)]++;
        }
        LOG_INFO("[P61] export mix:");
        for (const auto& kv : stat) LOG_INFO("[P61]   %s x%d", kv.first.c_str(), kv.second);
    }

    std::string error;
    if (success) {
        std::unique_ptr<StreamingModelWriter> writer;
        if (output_is_safetensors) {
            writer = std::make_unique<SafetensorsStreamingWriter>();
        } else {
            writer = std::make_unique<GGUFStreamingWriter>();
        }
        success = write_model_file_streaming(model_loader, output_path, tensors, *writer, n_threads, convert_name, mem, &error);
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    return success;
}

static bool has_active_loras(const sd_lora_t* loras, int lora_count) {
    if (loras == nullptr) {
        return false;
    }
    for (int i = 0; i < lora_count; i++) {
        if (loras[i].multiplier != 0.0f) {
            return true;
        }
    }
    return false;
}

static std::vector<ModelManager::LoraSpec> build_lora_specs(const sd_lora_t* loras, int lora_count) {
    std::vector<ModelManager::LoraSpec> specs;
    specs.reserve(lora_count);
    for (int i = 0; i < lora_count; i++) {
        ModelManager::LoraSpec spec;
        spec.path          = loras[i].path != nullptr ? loras[i].path : "";
        spec.multiplier    = loras[i].multiplier;
        spec.is_high_noise = loras[i].is_high_noise;
        spec.required      = true;
        specs.push_back(std::move(spec));
    }
    return specs;
}

static ModelComponent component_for_tensor_name(const std::string& name) {
    if (starts_with(name, "text_encoders.") || starts_with(name, "cond_stage_model.")) {
        return ModelComponent::Conditioner;
    }
    if (starts_with(name, "model.high_noise_diffusion_model.")) {
        return ModelComponent::HighNoiseDiffusion;
    }
    if (starts_with(name, "model.diffusion_model.")) {
        return ModelComponent::Diffusion;
    }
    if (starts_with(name, "vae.") || starts_with(name, "first_stage_model.")) {
        return ModelComponent::VAE;
    }
    return ModelComponent::Diffusion;
}

static bool load_model_into_memory(ModelLoader& model_loader,
                                   const sd_lora_t* loras,
                                   int lora_count,
                                   int n_threads,
                                   ModelManager& manager,
                                   ggml_backend_t cpu,
                                   ggml_context* ctx,
                                   std::map<std::string, ggml_tensor*>& mem,
                                   std::vector<ggml_tensor*>& all_tensors,
                                   ggml_type output_type,
                                   const TensorTypeRules& type_rules) {
    LOG_INFO("load_model_into_memory enter");
    const String2TensorStorage& storage_map = model_loader.get_tensor_storage_map();
    if (storage_map.empty()) {
        LOG_ERROR("no tensors to load into memory for convert");
        return false;
    }

    const SDVersion version  = model_loader.get_sd_version();
    ModelLoader merge_loader = model_loader;
    merge_loader.convert_tensors_name();

    std::map<ModelComponent, std::map<std::string, ggml_tensor*>> groups;
    size_t total_bytes = 0;
    for (const auto& [name, storage] : storage_map) {
        ggml_type load_type         = storage.type;
        const ggml_type export_type = get_export_tensor_type(model_loader, storage, output_type, type_rules);
        const bool quant_src        = ggml_is_quantized(storage.type);
        const bool quant_dst        = ggml_is_quantized(export_type);
        if (quant_src && quant_dst) {
            load_type = GGML_TYPE_F32;
        } else if (quant_src || quant_dst) {
            load_type = quant_src ? export_type : storage.type;
        } else if (storage.type == GGML_TYPE_F32 || export_type == GGML_TYPE_F32) {
            load_type = GGML_TYPE_F32;
        }
        // [P61] Cuantizar requiere cargar en un tipo NO cuantizado y convertir despues; no se
        // puede escribir a traves de un load_tensor en un destino ya Q8_0 (copiaria bytes F32).
        if (ggml_is_quantized(load_type) && load_type != storage.type) {
            load_type = GGML_TYPE_F32;
        }
        ggml_tensor* tensor = ggml_new_tensor(ctx, load_type, storage.n_dims, storage.ne);
        if (tensor == nullptr) {
            LOG_ERROR("failed to create tensor '%s' for in-memory convert", name.c_str());
            return false;
        }
        const std::string merge_name = convert_tensor_name(name, version);
        auto& group                  = groups[component_for_tensor_name(merge_name)];
        if (!group.emplace(merge_name, tensor).second) {
            LOG_ERROR("duplicate tensor name '%s' after name conversion", merge_name.c_str());
            return false;
        }
        mem[name] = tensor;
        all_tensors.push_back(tensor);
        total_bytes += ggml_nbytes(tensor);
    }
    LOG_INFO("loading %zu tensors (%.2f GB) into memory for convert",
             all_tensors.size(), total_bytes / (1024.0 * 1024.0 * 1024.0));

    manager.set_n_threads(n_threads);
    manager.set_enable_mmap(false);
    if (!manager.set_loader(std::move(merge_loader))) {
        LOG_ERROR("failed to set model loader for in-memory convert");
        return false;
    }

    for (const auto& [component, tensors] : groups) {
        if (!manager.register_param_tensors(component, tensors, ModelManager::ResidencyMode::ParamBackend, cpu, cpu)) {
            LOG_ERROR("failed to register %s tensors for in-memory convert", model_component_name(component));
            return false;
        }
    }

    std::vector<ModelManager::LoraSpec> lora_specs = build_lora_specs(loras, lora_count);
    if (!manager.prepare_lora_sources(lora_specs)) {
        LOG_ERROR("failed to prepare LoRA sources for convert");
        return false;
    }
    if (!manager.set_loras(lora_specs, version)) {
        LOG_ERROR("failed to set LoRAs for convert");
        return false;
    }
    if (!manager.prepare_params(all_tensors)) {
        LOG_ERROR("failed to load model into memory for convert");
        return false;
    }

    return true;
}

static bool convert_model_in_memory(ModelLoader& model_loader,
                                    const char* output_path,
                                    sd_type_t output_type,
                                    const char* tensor_type_rules,
                                    bool convert_name,
                                    int n_threads,
                                    const sd_lora_t* loras,
                                    int lora_count) {
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(sd_backend_cpu_init(), ggml_backend_free);
    if (cpu == nullptr) {
        LOG_ERROR("failed to init CPU backend for in-memory convert");
        return false;
    }

    ggml_init_params ctx_params;
    ctx_params.mem_size   = model_loader.get_tensor_storage_map().size() * ggml_tensor_overhead();
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = true;
    ggml_context* ctx     = ggml_init(ctx_params);
    if (ctx == nullptr) {
        LOG_ERROR("ggml_init failed for in-memory convert");
        return false;
    }

    ggml_type type             = sd_type_to_ggml_type(output_type);
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    std::vector<ggml_tensor*> all_tensors;
    std::map<std::string, ggml_tensor*> mem;
    bool success = false;
    {
        // The manager owns the buffers backing `mem` and must outlive the export; it is
        // destroyed (freeing those buffers) before the tensor context is freed below.
        ModelManager manager;
        if (load_model_into_memory(model_loader, loras, lora_count, n_threads, manager, cpu.get(), ctx, mem, all_tensors, type, type_rules)) {
            success = export_loaded_model(model_loader, output_path, output_type, tensor_type_rules, n_threads, convert_name, &mem);
        }
    }
    ggml_free(ctx);
    return success;
}

bool convert_with_components(const char* model_path,
                             const char* clip_l_path,
                             const char* clip_g_path,
                             const char* t5xxl_path,
                             const char* diffusion_model_path,
                             const char* vae_path,
                             const char* output_path,
                             sd_type_t output_type,
                             const char* tensor_type_rules,
                             bool convert_name,
                             int n_threads,
                             const sd_lora_t* loras,
                             int lora_count) {
    if (!validate_tensor_types(output_type, tensor_type_rules)) {
        return false;
    }

    if (loras != nullptr) {
        for (int i = 0; i < lora_count; i++) {
            LOG_INFO("lora %d: '%s' (multiplier %.2f%s)",
                     i + 1, loras[i].path != nullptr ? loras[i].path : "",
                     loras[i].multiplier,
                     loras[i].is_high_noise ? ", high noise" : "");
        }
    }

    ModelLoader model_loader;
    bool loaded_any = false;

    if (!init_convert_path(model_loader, model_path, "", loaded_any) ||
        !init_convert_path(model_loader, clip_l_path, "text_encoders.clip_l.transformer.", loaded_any) ||
        !init_convert_path(model_loader, clip_g_path, "text_encoders.clip_g.transformer.", loaded_any) ||
        !init_convert_path(model_loader, t5xxl_path, "text_encoders.t5xxl.transformer.", loaded_any) ||
        !init_convert_path(model_loader, diffusion_model_path, "model.diffusion_model.", loaded_any) ||
        !init_convert_path(model_loader, vae_path, "vae.", loaded_any)) {
        return false;
    }

    if (!loaded_any) {
        LOG_ERROR("no input model path provided for convert");
        return false;
    }

    if (has_active_loras(loras, lora_count)) {
        // [P61] Solo LoRA exige materializar el modelo en RAM. En ese camino va el
        // riesgo del aborto 0xC0000409 (load en F32 vs lo que espera
        // ModelManager::prepare_params) - ver doc ENTREGA-senior-CONVERSION-Q8 §6.
        LOG_INFO("loading the full model into memory to apply LoRAs");
        return convert_model_in_memory(model_loader, output_path, output_type, tensor_type_rules, convert_name, n_threads,
                                       loras, lora_count);
    }

    if (convert_name) {
        model_loader.convert_tensors_name();
    }

    // [P61] Streaming cuando no hay LoRA: load_tensor_for_export carga en el tipo
    // origen y convierte con convert_tensor (datos ya en el tipo destino), asi que
    // el fichero sale correcto sin materializar 4 GB en RAM. El modo en memoria
    // aborta con override de tipo (0xC0000005/0xC0000409).
    return export_loaded_model(model_loader,
                               output_path,
                               output_type,
                               tensor_type_rules,
                               n_threads,
                               convert_name);
}

bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name) {
    return convert_with_components(input_path,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   vae_path,
                                   output_path,
                                   output_type,
                                   tensor_type_rules,
                                   convert_name,
                                   0,
                                   nullptr,
                                   0);
}
