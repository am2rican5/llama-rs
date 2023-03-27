use std::io::BufRead;
use std::io::Write;

use cli_args::CLI_ARGS;
use llama_rs::{InferenceError, InferenceParameters};

mod cli_args;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = &*CLI_ARGS;
    let mut embedding_parameters = InferenceParameters::default();
    embedding_parameters.n_threads = args.num_threads as i32;
    embedding_parameters.n_batch = args.batch_size;

    let(model, vocab) = 
    llama_rs::Model::load(&args.model_path, args.num_ctx_tokens as i32, |progress| {
        use llama_rs::LoadProgress;
        match progress {
            LoadProgress::HyperparametersLoaded(hparams) => {
                log::debug!("Loaded HyperParams {hparams:#?}")
            }
            LoadProgress::BadToken { index } => {
                log::info!("Warning: Bad token in vocab at index {index}")
            }
            LoadProgress::ContextSize { bytes } => log::info!(
                "ggml ctx size = {:.2} MB\n",
                bytes as f64 / (1024.0 * 1024.0)
            ),
            LoadProgress::MemorySize { bytes, n_mem } => log::info!(
                "Memory size: {} MB {}",
                bytes as f32 / 1024.0 / 1024.0,
                n_mem
            ),
            LoadProgress::PartLoading {
                file,
                current_part,
                total_parts,
            } => log::info!(
                "Loading model part {}/{} from '{}'\n",
                current_part,
                total_parts,
                file.to_string_lossy(),
            ),
            LoadProgress::PartTensorLoaded {
                current_tensor,
                tensor_count,
                ..
            } => {
                if current_tensor % 8 == 0 {
                    log::info!("Loaded tensor {current_tensor}/{tensor_count}");
                }
            }
            LoadProgress::PartLoaded {
                file,
                byte_size,
                tensor_count,
            } => {
                log::info!("Loading of '{}' complete", file.to_string_lossy());
                log::info!(
                    "Model size = {:.2} MB / num tensors = {}",
                    byte_size as f64 / 1024.0 / 1024.0,
                    tensor_count
                );
            }
        }
    })
    .expect("Could not load model");

    log::info!("Model fully loaded!");


    let prompts = if let Some(path) = &args.prompt_file {
        let file = std::fs::File::open(path).expect("Could not open prompt file");
        let reader = std::io::BufReader::new(file);

        reader
            .lines()
            .map(|line| line.expect("Could not read line from prompt file"))
            .collect()
    } else if let Some(prompt) = &args.prompt {
        vec![prompt.clone()]
    } else {
        log::error!("No prompt or prompt file was provided. See --help");
        vec!["The quick brown fox jumps over the lazy dog.".to_string()]
    };

    let mut output_file = std::fs::File::create(&args.output_file).expect("Could not open output file");

    for prompt in prompts {
        let mut session = model.start_session(0);

        let res = session.embed_prompt(
            &model,
            &vocab,
            &embedding_parameters,
            &prompt,
        );

        match res {
            Ok(stats) => {
                println!("{}", stats);
                if let Err(e) = writeln!(output_file, "{}", stats.vector_to_string()) {
                    log::error!("Could not write to output file: {}", e);
                }
            }
            Err(llama_rs::InferenceError::ContextFull) => {
                log::warn!("Context window full, stopping inference.")
            }
            Err(InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
        }
    }
}
