use clap::Parser;
use once_cell::sync::Lazy;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Where to load the model path from
    #[arg(long, short = 'm')]
    pub model_path: String,

    /// The prompt to feed the generator
    #[arg(long, short = 'p', default_value = None)]
    pub prompt: Option<String>,

    /// A file to read the prompt from. Takes precedence over `prompt` if set.
    #[arg(long, short = 'f', default_value = None)]
    pub prompt_file: Option<String>,

    /// Output file to write the generated text to. If not set, the output will be printed to stdout.
    #[arg(long, short = 'o', default_value = None)]
    pub output_file: Option<String>,

    /// Sets the number of threads to use
    #[arg(long, short = 't', default_value_t = num_cpus::get_physical())]
    pub num_threads: usize,

    /// Sets the size of the context (in tokens). Allows feeding longer prompts.
    /// Note that this affects memory. TODO: Unsure how large the limit is.
    #[arg(long, default_value_t = 512)]
    pub num_ctx_tokens: usize,

    /// How many tokens from the prompt at a time to feed the network. Does not
    /// affect generation.
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// Specifies the seed to use during sampling. Note that, depending on
    /// hardware, the same seed may lead to different results on two separate
    /// machines.
    #[arg(long, default_value = None)]
    pub seed: Option<u64>,
}

/// CLI args are stored in a lazy static variable so they're accessible from
/// everywhere. Arguments are parsed on first access.
pub static CLI_ARGS: Lazy<Args> = Lazy::new(Args::parse);
