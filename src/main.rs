use anyhow::{anyhow, Context, Result};
use clap::Parser;
use crossterm::{
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

// ─── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "zeno_chat")]
#[command(about = "Two LLMs chatting with each other using llama.cpp")]
struct Args {
    #[arg(short, long)] prompt: Option<String>,
    #[arg(short, long)] limit:  Option<String>,
}

// ─── CONFIG DEFAULTS ─────────────────────────────────────────────────────────

fn default_agent_name_a()    -> String { "Agent A".to_string() }
fn default_agent_name_b()    -> String { "Agent B".to_string() }
fn default_agent_color_a()   -> String { "DarkYellow".to_string() }
fn default_agent_color_b()   -> String { "Blue".to_string() }
fn default_header_color()    -> String { "DarkGrey".to_string() }
fn default_stats_color()     -> String { "Grey".to_string() }
fn default_thinking_color()  -> String { "DarkGrey".to_string() }
fn default_log_mode()        -> String { "json".to_string() }
fn default_log_file()        -> String { "chat_log.jsonl".to_string() }
fn default_limit()           -> String { "20".to_string() }
fn default_show_stats()      -> bool   { true }
fn default_log_stats()       -> bool   { false }
fn default_show_thinking()   -> bool   { false }
fn default_log_thinking()    -> bool   { false }
fn default_log_debug()       -> bool   { false }
fn default_compress_thresh() -> usize  { 30 }
fn default_compress_keep()   -> usize  { 20 }

// ─── MODEL PARAMS ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ModelParams {
    ctx: usize,
    temp: f32,
    n_predict: usize,
    /// Override auto-detected GPU layers. null = use auto-detected value.
    #[serde(default)]
    n_gpu_layers_override: Option<usize>,
    /// Passed as --reasoning-format to llama.cpp. "auto" skips the flag.
    reasoning: String,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self { ctx: 4096, temp: 0.8, n_predict: 512, n_gpu_layers_override: None, reasoning: "auto".to_string() }
    }
}

// ─── APP CONFIG ──────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
struct AppConfig {
    selected_model: String,

    // Agent identity
    #[serde(default = "default_agent_name_a")] agent_name_a: String,
    #[serde(default = "default_agent_name_b")] agent_name_b: String,
    system_prompt_a: String,
    system_prompt_b: String,
    start_prompt: String,
    #[serde(default = "default_limit")] limit: String,

    // Display colors (string names: Black/Red/Green/Yellow/Blue/Magenta/Cyan/White/
    // Dark{Red,Green,Yellow,Blue,Magenta,Cyan}/Grey/DarkGrey or rgb(r,g,b))
    #[serde(default = "default_agent_color_a")]  agent_color_a:  String,
    #[serde(default = "default_agent_color_b")]  agent_color_b:  String,
    #[serde(default = "default_header_color")]   header_color:   String,
    #[serde(default = "default_stats_color")]    stats_color:    String,
    #[serde(default = "default_thinking_color")] thinking_color: String,

    // Logging
    #[serde(default = "default_log_file")]    log_file:     String,
    #[serde(default = "default_log_mode")]    log_mode:     String, // "json" or "txt"
    #[serde(default = "default_show_stats")]  show_stats:   bool,
    #[serde(default = "default_log_stats")]   log_stats:    bool,
    #[serde(default = "default_show_thinking")] show_thinking: bool,
    #[serde(default = "default_log_thinking")] log_thinking: bool,
    #[serde(default = "default_log_debug")]   log_debug:    bool,

    // Context compression
    #[serde(default = "default_compress_thresh")] compress_threshold: usize,
    #[serde(default = "default_compress_keep")]   compress_keep:      usize,

    models: std::collections::HashMap<String, ModelParams>,
}

impl Default for AppConfig {
    fn default() -> Self {
        let mut models = std::collections::HashMap::new();
        models.insert("Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string(), ModelParams::default());
        Self {
            selected_model:  "Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string(),
            agent_name_a:    default_agent_name_a(),
            agent_name_b:    default_agent_name_b(),
            system_prompt_a: "You are Agent A. You are having a conversation with Agent B. Keep your responses short.".to_string(),
            system_prompt_b: "You are Agent B. You are having a conversation with Agent A. Keep your responses short.".to_string(),
            start_prompt:    "Agent A: Hello! What shall we talk about today?".to_string(),
            limit:           default_limit(),
            agent_color_a:   default_agent_color_a(),
            agent_color_b:   default_agent_color_b(),
            header_color:    default_header_color(),
            stats_color:     default_stats_color(),
            thinking_color:  default_thinking_color(),
            log_file:        default_log_file(),
            log_mode:        default_log_mode(),
            show_stats:      default_show_stats(),
            log_stats:       default_log_stats(),
            show_thinking:   default_show_thinking(),
            log_thinking:    default_log_thinking(),
            log_debug:       default_log_debug(),
            compress_threshold: default_compress_thresh(),
            compress_keep:      default_compress_keep(),
            models,
        }
    }
}

// ─── COLOR PARSING ───────────────────────────────────────────────────────────

fn parse_color(s: &str) -> Color {
    match s.to_lowercase().replace('_', "").as_str() {
        "black"                         => Color::Black,
        "red"                           => Color::Red,
        "green"                         => Color::Green,
        "yellow"                        => Color::Yellow,
        "blue"                          => Color::Blue,
        "magenta"                       => Color::Magenta,
        "cyan"                          => Color::Cyan,
        "white"                         => Color::White,
        "darkred"                       => Color::DarkRed,
        "darkgreen"                     => Color::DarkGreen,
        "darkyellow" | "orange"         => Color::DarkYellow,
        "darkblue"                      => Color::DarkBlue,
        "darkmagenta"                   => Color::DarkMagenta,
        "darkcyan"                      => Color::DarkCyan,
        "grey"  | "gray"                => Color::Grey,
        "darkgrey" | "darkgray"         => Color::DarkGrey,
        other => {
            // Support "rgb(r,g,b)" or "r,g,b"
            let s = other.trim_start_matches("rgb(").trim_end_matches(')');
            let p: Vec<&str> = s.split(',').collect();
            if p.len() == 3 {
                if let (Ok(r), Ok(g), Ok(b)) = (
                    p[0].trim().parse::<u8>(), p[1].trim().parse::<u8>(), p[2].trim().parse::<u8>()
                ) { return Color::Rgb { r, g, b }; }
            }
            Color::White // fallback
        }
    }
}

// ─── DATA TYPES ──────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct GithubRelease { tag_name: String, assets: Vec<GithubAsset> }
#[derive(Deserialize, Debug)]
struct GithubAsset { name: String, browser_download_url: String }

#[derive(Clone, Copy, Debug, PartialEq)]
enum Role { #[allow(dead_code)] System, User, Assistant }

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Role::System => write!(f, "system"), Role::User => write!(f, "user"), Role::Assistant => write!(f, "assistant") }
    }
}

#[derive(Clone, Debug)]
struct Message { role: Role, content: String }

// ─── LOGGING ─────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatLogEntry<'a> {
    timestamp:                                    String,
    speaker:                                      String,
    text:                                         String,
    #[serde(skip_serializing_if = "Option::is_none")] thinking: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")] stats:    Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")] debug:    Option<&'a str>,
}

fn get_timestamp() -> String {
    if let Ok(now) = time::OffsetDateTime::now_local() {
        format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", now.year(), u8::from(now.month()), now.day(), now.hour(), now.minute(), now.second())
    } else {
        let now = time::OffsetDateTime::now_utc();
        format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC", now.year(), u8::from(now.month()), now.day(), now.hour(), now.minute(), now.second())
    }
}

fn append_to_log(
    log_path: &Path, speaker: &str, text: &str,
    thinking: Option<&str>, stats: Option<&str>, debug: Option<&str>,
    log_mode: &str,
) -> Result<()> {
    let timestamp = get_timestamp();
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(log_path)?;
    if log_mode.to_lowercase() == "txt" {
        writeln!(file, "[{}] {}: {}", timestamp, speaker, text)?;
        if let Some(t) = thinking { writeln!(file, "  <thinking> {}", t)?; }
        if let Some(s) = stats    { writeln!(file, "  [stats] {}",    s)?; }
        if let Some(d) = debug    { writeln!(file, "  [debug]\n{}",   d)?; }
    } else {
        let entry = ChatLogEntry { timestamp, speaker: speaker.to_string(), text: text.to_string(), thinking, stats, debug };
        writeln!(file, "{}", serde_json::to_string(&entry)?)?;
    }
    Ok(())
}

// ─── DOWNLOAD HELPERS ────────────────────────────────────────────────────────

fn download_file_with_progress(client: &reqwest::blocking::Client, url: &str, dest_path: &Path, message: &str) -> Result<()> {
    if let Some(p) = dest_path.parent() { std::fs::create_dir_all(p)?; }
    let mut resp = client.get(url).send()?;
    if !resp.status().is_success() { return Err(anyhow!("HTTP {} for {}", resp.status(), url)); }
    let total = resp.content_length().ok_or_else(|| anyhow!("No content-length for {}", url))?;
    let pb = ProgressBar::new(total);
    pb.set_style(ProgressStyle::default_bar().template("{msg}\n[{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")?.progress_chars("#>-"));
    pb.set_message(message.to_string());
    let mut file = File::create(dest_path)?;
    let mut buf = [0u8; 8192];
    let mut done = 0u64;
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 { break; }
        file.write_all(&buf[..n])?;
        done += n as u64;
        pb.set_position(done);
    }
    pb.finish_with_message(format!("Downloaded {}", dest_path.file_name().unwrap().to_string_lossy()));
    Ok(())
}

fn extract_zip(zip_path: &Path, target_dir: &Path) -> Result<()> {
    let f = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(f)?;
    std::fs::create_dir_all(target_dir)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let outpath = match entry.enclosed_name() { Some(p) => target_dir.join(p), None => continue };
        if entry.name().ends_with('/') { std::fs::create_dir_all(&outpath)?; }
        else {
            if let Some(p) = outpath.parent() { if !p.exists() { std::fs::create_dir_all(p)?; } }
            std::io::copy(&mut entry, &mut std::fs::File::create(&outpath)?)?;
        }
    }
    Ok(())
}

fn setup(client: &reqwest::blocking::Client, selected_model: &str) -> Result<(String, usize)> {
    let base       = std::env::current_exe()?.parent().context("no exe dir")?.to_path_buf();
    let model_path = base.join("gguf").join(selected_model);
    let vulkan_exe = base.join("llama_bin").join("vulkan").join("llama-completion.exe");
    let cpu_exe    = base.join("llama_bin").join("cpu").join("llama-completion.exe");

    if !model_path.exists() {
        if selected_model == "Llama-3.2-1B-Instruct-Q4_K_M.gguf" {
            println!("Downloading default model (~648 MB)...");
            download_file_with_progress(client,
                "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                &model_path, "Downloading model")?;
        } else { return Err(anyhow!("Model '{}' not found in gguf/", selected_model)); }
    }

    if !vulkan_exe.exists() || !cpu_exe.exists() {
        println!("Setting up llama.cpp binaries...");
        let mut vu: Option<String> = None;
        let mut cu: Option<String> = None;
        if let Ok(r) = client.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest").send() {
            if r.status().is_success() {
                if let Ok(rel) = r.json::<GithubRelease>() {
                    println!("Found llama.cpp release: {}", rel.tag_name);
                    for a in rel.assets {
                        if a.name.contains("bin-win-vulkan-x64.zip") { vu = Some(a.browser_download_url); }
                        else if a.name.contains("bin-win-cpu-x64.zip") { cu = Some(a.browser_download_url); }
                    }
                }
            }
        }
        let vd = base.join("llama_bin").join("vulkan");
        let cd = base.join("llama_bin").join("cpu");
        let vu = vu.unwrap_or_else(|| "https://github.com/ggml-org/llama.cpp/releases/download/b9789/llama-b9789-bin-win-vulkan-x64.zip".to_string());
        let cu = cu.unwrap_or_else(|| "https://github.com/ggml-org/llama.cpp/releases/download/b9789/llama-b9789-bin-win-cpu-x64.zip".to_string());
        if !vulkan_exe.exists() { let z = vd.join("vulkan.zip"); download_file_with_progress(client, &vu, &z, "Downloading Vulkan")?; println!("Extracting Vulkan..."); extract_zip(&z, &vd)?; let _ = std::fs::remove_file(z); }
        if !cpu_exe.exists()   { let z = cd.join("cpu.zip");    download_file_with_progress(client, &cu, &z, "Downloading CPU")?;    println!("Extracting CPU...");    extract_zip(&z, &cd)?; let _ = std::fs::remove_file(z); }
    }

    let use_vulkan = Command::new(&vulkan_exe).arg("-h").stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
    if use_vulkan {
        println!("Vulkan GPU acceleration enabled.");
        Ok((vulkan_exe.to_string_lossy().to_string(), 99))
    } else {
        use crossterm::style::Stylize;
        println!("{}", "Vulkan unavailable, falling back to CPU.".yellow());
        Ok((cpu_exe.to_string_lossy().to_string(), 0))
    }
}

// ─── PROMPT FORMATTING ───────────────────────────────────────────────────────

fn format_prompt(sys: &str, summary: Option<&str>, history: &[Message]) -> String {
    let mut p = format!("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", sys);
    if let Some(s) = summary {
        p.push_str(&format!("<|start_header_id|>system<|end_header_id|>\n\n[Summary of previous conversation: {}]<|eot_id|>", s));
    }
    for m in history {
        p.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", m.role, m.content));
    }
    p.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    p
}

// ─── STATS ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct DetailedStats {
    eval_time_ms: f32, eval_tokens: usize, eval_speed: f32,
    gen_time_ms: f32,  gen_tokens: usize,  gen_speed: f32,
    sampling_time_ms: f32, sampling_runs: usize, sampling_speed: f32,
    load_time_ms: f32, total_time_ms: f32,
}

fn parse_detailed_stats(log: &str) -> Option<DetailedStats> {
    let mut s = DetailedStats {
        eval_time_ms: 0.0, eval_tokens: 0, eval_speed: 0.0,
        gen_time_ms: 0.0,  gen_tokens: 0,  gen_speed: 0.0,
        sampling_time_ms: 0.0, sampling_runs: 0, sampling_speed: 0.0,
        load_time_ms: 0.0, total_time_ms: 0.0,
    };
    let pv = |line: &str, label: &str, end: &str| -> Option<f32> {
        let pos = line.find(label)?;
        line[pos + label.len()..][..line[pos + label.len()..].find(end)?].trim().parse::<f32>().ok()
    };
    for line in log.lines() {
        if line.contains("prompt eval time") {
            if let Some(v) = pv(line, "prompt eval time =", "ms") { s.eval_time_ms = v; }
            if let Some(v) = pv(line, "/", "tokens")               { s.eval_tokens  = v as usize; }
            if let Some(v) = pv(line, ",", "tokens per second")    { s.eval_speed   = v; }
        } else if line.contains("eval time") && !line.contains("prompt eval time") {
            if let Some(v) = pv(line, "eval time =", "ms") { s.gen_time_ms = v; }
            if let Some(v) = pv(line, "/", "runs")          { s.gen_tokens  = v as usize; }
            if let Some(v) = pv(line, ",", "tokens per second") { s.gen_speed = v; }
        } else if line.contains("sample time") {
            if let Some(v) = pv(line, "sample time =", "ms") { s.sampling_time_ms  = v; }
            if let Some(v) = pv(line, "/", "runs")            { s.sampling_runs     = v as usize; }
            if let Some(v) = pv(line, ",", "tokens per second") { s.sampling_speed  = v; }
        } else if line.contains("load time")  { if let Some(v) = pv(line, "load time =", "ms")  { s.load_time_ms  = v; } }
          else if line.contains("total time") { if let Some(v) = pv(line, "total time =", "ms") { s.total_time_ms = v; } }
    }
    Some(s)
}

fn format_stats(s: &DetailedStats) -> String {
    format!(
        "eval {:.1}t/s ({:.0}ms/{et}t) | gen {:.1}t/s ({:.0}ms/{gt}t) | samp {:.1}t/s ({:.0}ms/{sr}r) | load {:.0}ms | total {:.0}ms",
        s.eval_speed, s.eval_time_ms, s.gen_speed, s.gen_time_ms,
        s.sampling_speed, s.sampling_time_ms, s.load_time_ms, s.total_time_ms,
        et = s.eval_tokens, gt = s.gen_tokens, sr = s.sampling_runs
    )
}

// ─── THINK TAG SPLITTER ──────────────────────────────────────────────────────

/// Splits the raw LLM output into (normal_text, thinking_text), stripping the tags.
#[allow(dead_code)]
fn split_thinking(raw: &str) -> (String, String) {
    let mut normal = String::new();
    let mut think  = String::new();
    let mut in_think = false;
    let mut i = 0;
    while i < raw.len() {
        let rem = &raw[i..];
        if !in_think && rem.starts_with("<think>") { in_think = true;  i += 7; }
        else if in_think && rem.starts_with("</think>") { in_think = false; i += 8; }
        else {
            let ch = rem.chars().next().unwrap();
            if in_think { think.push(ch); } else { normal.push(ch); }
            i += ch.len_utf8();
        }
    }
    (normal.trim().to_string(), think.trim().to_string())
}

// ─── STREAM DISPLAY ──────────────────────────────────────────────────────────

const EOT_PATTERNS: &[&str] = &[" [end of text]", "[end of text]", "<|eot_id|>"];

/// Handles real-time streaming display with:
/// - EOT pattern filtering
/// - Think tag detection (colorised or hidden based on config)
/// - Trailing newline suppression (no blank lines before stats)
/// - Mid-stream double blank line cap (max 1 blank line inside a message)
struct StreamDisplay {
    // EOT filtering
    eot_buf: Vec<char>,
    // Think tag detection
    tag_buf: String,
    in_think: bool,
    // Trailing newline suppression
    pending_nl: usize,
    // Config
    msg_color:     Color,
    thinking_color: Color,
    show_thinking: bool,
    // Output accumulation (separated)
    pub normal_buf: String,
    pub think_buf:  String,
}

impl StreamDisplay {
    fn new(msg_color: Color, thinking_color: Color, show_thinking: bool) -> Self {
        let _ = execute!(std::io::stdout(), SetForegroundColor(msg_color));
        let _ = std::io::stdout().flush();
        Self {
            eot_buf: Vec::new(), tag_buf: String::new(), in_think: false,
            pending_nl: 0, msg_color, thinking_color, show_thinking,
            normal_buf: String::new(), think_buf: String::new(),
        }
    }

    /// Feed one character from the raw LLM output.
    fn push(&mut self, ch: char) {
        // Stage 1: EOT filtering
        self.eot_buf.push(ch);
        let qs: String = self.eot_buf.iter().collect();
        if EOT_PATTERNS.iter().any(|p| **p == qs) { self.eot_buf.clear(); return; }
        if EOT_PATTERNS.iter().any(|p| p.starts_with(qs.as_str()) && qs.len() < p.len()) { return; }
        // Mismatch: flush eot_buf through stage 2
        let chars: Vec<char> = self.eot_buf.drain(..).collect();
        for c in chars { self.think_stage(c); }
    }

    fn think_stage(&mut self, ch: char) {
        // Stage 2: think tag detection — only buffer when we see '<'
        if ch == '<' || !self.tag_buf.is_empty() {
            self.tag_buf.push(ch);
            let tb = self.tag_buf.clone();

            if !self.in_think && tb.ends_with("<think>") {
                // Flush content before tag as normal, enter thinking mode
                let before = tb[..tb.len()-7].to_string();
                self.tag_buf.clear();
                self.in_think = true;
                for c in before.chars() { self.emit_normal(c); }
                if self.show_thinking {
                    let _ = execute!(std::io::stdout(), SetForegroundColor(self.thinking_color));
                }
                return;
            }
            if self.in_think && tb.ends_with("</think>") {
                // Flush content before end-tag as thinking, exit thinking mode
                let content = tb[..tb.len()-8].to_string();
                self.tag_buf.clear();
                self.in_think = false;
                for c in content.chars() { self.emit_think(c); }
                let _ = execute!(std::io::stdout(), SetForegroundColor(self.msg_color));
                return;
            }
            // Check if still a valid prefix of a tag
            let open_prefix  = !self.in_think && "<think>".starts_with(tb.as_str());
            let close_prefix =  self.in_think && "</think>".starts_with(tb.as_str());
            if open_prefix || close_prefix { return; }
            // Not a tag — flush all buffered chars
            let content = self.tag_buf.clone(); self.tag_buf.clear();
            for c in content.chars() {
                if self.in_think { self.emit_think(c); } else { self.emit_normal(c); }
            }
        } else {
            if self.in_think { self.emit_think(ch); } else { self.emit_normal(ch); }
        }
    }

    fn emit_normal(&mut self, ch: char) {
        if ch == '\n' {
            self.pending_nl += 1;
        } else {
            // Flush buffered newlines, capped at 2 (= max 1 blank line mid-message)
            let to_flush = self.pending_nl.min(2);
            if to_flush > 0 {
                let nl = "\n".repeat(to_flush);
                let _ = execute!(std::io::stdout(), Print(&nl));
                self.normal_buf.push_str(&nl);
                self.pending_nl = 0;
            }
            let _ = execute!(std::io::stdout(), Print(ch));
            self.normal_buf.push(ch);
        }
    }

    fn emit_think(&mut self, ch: char) {
        self.think_buf.push(ch);
        if self.show_thinking {
            let _ = execute!(std::io::stdout(), Print(ch));
            if ch == '\n' { let _ = std::io::stdout().flush(); }
        }
    }

    /// Call after the LLM process exits. Flushes remaining buffers, discards trailing newlines.
    fn finish(&mut self) {
        let remaining: Vec<char> = self.eot_buf.drain(..).collect();
        for c in remaining { self.think_stage(c); }
        let remaining = self.tag_buf.clone(); self.tag_buf.clear();
        for c in remaining.chars() {
            if self.in_think { self.emit_think(c); } else { self.emit_normal(c); }
        }
        // Trailing newlines: discard from display AND from normal_buf
        self.pending_nl = 0;
        let _ = execute!(std::io::stdout(), ResetColor);
        let _ = std::io::stdout().flush();
    }
}

// ─── RUN AND STREAM ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_and_stream(
    binary_path:   &str,
    model_path:    &str,
    prompt:        &str,
    auto_ngl:      usize,
    msg_color:     Color,
    header_color:  Color,
    stats_color:   Color,
    speaker:       &str,
    params:        &ModelParams,
    show_stats:    bool,
    show_thinking: bool,
    thinking_color: Color,
) -> Result<(String, String, Option<DetailedStats>, String)> {
    // Returns: (normal_response, think_content, stats, stderr_debug)
    let mut stdout = std::io::stdout();

    // Print ##Speaker## header in header_color
    let _ = execute!(stdout, SetForegroundColor(header_color), Print(format!("\n##{speaker}##\n")), ResetColor);
    let _ = stdout.flush();

    let ngl = params.n_gpu_layers_override.unwrap_or(auto_ngl);

    let mut cmd = Command::new(binary_path);
    cmd.arg("-m").arg(model_path)
       .arg("-p").arg(prompt)
       .arg("-n").arg(params.n_predict.to_string())
       .arg("-c").arg(params.ctx.to_string())
       .arg("--temp").arg(params.temp.to_string())
       .arg("-ngl").arg(ngl.to_string())
       .arg("-no-cnv")
       .arg("--no-display-prompt")
       .arg("--color").arg("off");
    // Pass reasoning-format flag when explicitly set
    if params.reasoning.to_lowercase() != "auto" {
        cmd.arg("--reasoning-format").arg(&params.reasoning);
    }

    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    let mut proc_stdout = child.stdout.take().context("no stdout")?;
    let mut proc_stderr = child.stderr.take().context("no stderr")?;

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = String::new();
        if proc_stderr.read_to_string(&mut buf).is_ok() { let _ = tx.send(buf); }
    });

    let mut display = StreamDisplay::new(msg_color, thinking_color, show_thinking);
    let mut byte_buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 1024];

    loop {
        let n = proc_stdout.read(&mut tmp)?;
        if n == 0 { break; }
        byte_buf.extend_from_slice(&tmp[..n]);
        while !byte_buf.is_empty() {
            let s = match std::str::from_utf8(&byte_buf) {
                Ok(v) => { let s = v.to_string(); byte_buf.clear(); s }
                Err(e) => {
                    let up = e.valid_up_to();
                    if up == 0 { if e.error_len().is_none() { break; } else { byte_buf.drain(..1); continue; } }
                    let s = std::str::from_utf8(&byte_buf[..up]).unwrap().to_string();
                    byte_buf.drain(..up); s
                }
            };
            for ch in s.chars() { display.push(ch); }
            let _ = std::io::stdout().flush();
        }
    }
    display.finish();

    let _ = child.wait()?;
    let stderr_log = rx.recv().unwrap_or_default();
    let stats = parse_detailed_stats(&stderr_log);

    // Print stats immediately after message (no blank line between)
    if show_stats {
        if let Some(ref s) = stats {
            let _ = execute!(stdout, SetForegroundColor(stats_color), Print(format!("[{}]\n", format_stats(s))), ResetColor);
        }
    }
    // Blank line to separate entries
    println!();

    Ok((display.normal_buf.trim().to_string(), display.think_buf.trim().to_string(), stats, stderr_log))
}

// ─── SUMMARIZE & COMPRESS ────────────────────────────────────────────────────

fn run_summarize(binary_path: &str, model_path: &str, prompt: &str, ngl: usize, params: &ModelParams) -> Result<String> {
    let out = Command::new(binary_path)
        .arg("-m").arg(model_path).arg("-p").arg(prompt)
        .arg("-n").arg("150").arg("-c").arg(params.ctx.to_string())
        .arg("--temp").arg(params.temp.to_string())
        .arg("-ngl").arg(ngl.to_string())
        .arg("-no-cnv").arg("--no-display-prompt").arg("--color").arg("off")
        .output()?;
    if !out.status.success() { return Err(anyhow!("Summarizer exited non-zero")); }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn compress_history(
    binary_path: &str, model_path: &str, ngl: usize,
    summary: &mut Option<String>, history: &mut Vec<Message>,
    self_name: &str, other_name: &str,
    params: &ModelParams, threshold: usize, keep: usize,
) -> Result<()> {
    if history.len() <= threshold { return Ok(()); }
    let drain_n = history.len().saturating_sub(keep);
    let mut transcript = String::new();
    if let Some(s) = &summary { transcript.push_str(&format!("[Previous Summary]: {}\n\n", s)); }
    for msg in &history[..drain_n] {
        let spk = match msg.role { Role::User => other_name, Role::Assistant => self_name, Role::System => "System" };
        transcript.push_str(&format!("{}: {}\n\n", spk, msg.content));
    }
    let sum_prompt = format_prompt(
        "You are a helpful assistant. Summarize the conversation concisely under 150 words without intro or outro.", None,
        &[Message { role: Role::User, content: format!("Summarize:\n\n{}", transcript) }]);
    if let Ok(new_sum) = run_summarize(binary_path, model_path, &sum_prompt, ngl, params) {
        *summary = Some(new_sum);
        history.drain(..drain_n);
    }
    Ok(())
}

fn clean_response(text: &str) -> String {
    text.replace(" [end of text]", "").replace("[end of text]", "")
        .replace("<|eot_id|>", "").replace("<think>", "").replace("</think>", "")
        .trim().to_string()
}

// ─── MAIN ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let _args = Args::parse();
    let base         = std::env::current_exe()?.parent().context("no exe dir")?.to_path_buf();
    // Load configuration from src/config.json instead of embedding it in the executable directory.
    let config_path  = base.join("src").join("config.json");

    let mut config: AppConfig = if config_path.exists() {
        match std::fs::read_to_string(&config_path).map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<AppConfig>(&s).map_err(|e| e.to_string()))
        {
            Ok(c) => c,
            Err(e) => { println!("[Warning: config error: {}. Using defaults.]", e); AppConfig::default() }
        }
    } else {
        let d = AppConfig::default();
        if let Ok(s) = serde_json::to_string_pretty(&d) { let _ = std::fs::write(&config_path, &s); }
        d
    };

    let model_params = config.models.entry(config.selected_model.clone()).or_insert_with(ModelParams::default).clone();
    // Persist config back (adds any new fields with defaults)
    if let Ok(s) = serde_json::to_string_pretty(&config) { let _ = std::fs::write(&config_path, s); }

    let client = reqwest::blocking::Client::builder().user_agent("zeno-chat-cli").build()?;
    let (binary_path, auto_ngl) = setup(&client, &config.selected_model)?;

    let model_path_str = base.join("gguf").join(&config.selected_model).to_string_lossy().to_string();
    let log_file_path  = base.join(&config.log_file);

    // Parse colors once
    let color_a       = parse_color(&config.agent_color_a);
    let color_b       = parse_color(&config.agent_color_b);
    let header_color  = parse_color(&config.header_color);
    let stats_color   = parse_color(&config.stats_color);
    let think_color   = parse_color(&config.thinking_color);

    let limit_str = config.limit.clone();
    let limit: Option<usize> = match limit_str.to_lowercase().trim() {
        "infinite" | "inf" | "none" => None,
        s => Some(s.parse::<usize>().unwrap_or(20)),
    };

    // Startup banner
    let mut stdout = std::io::stdout();
    let _ = execute!(stdout, SetForegroundColor(Color::DarkGrey),
        Print(format!("Zeno Chat  |  model: {}  |  limit: {}  |  log: {}\n",
            config.selected_model, limit_str, config.log_file)),
        ResetColor);
    let _ = execute!(stdout, SetForegroundColor(Color::Grey),
        Print(format!("Start: {}\n", config.start_prompt)),
        ResetColor);

    let mut history_a = vec![Message { role: Role::User, content: config.start_prompt.clone() }];
    let mut history_b = vec![Message { role: Role::User, content: config.start_prompt.clone() }];
    let mut summary_a: Option<String> = None;
    let mut summary_b: Option<String> = None;
    let mut turn = 0usize;

    loop {
        if limit.map(|l| turn >= l).unwrap_or(false) { break; }

        // ── Agent A ──────────────────────────────────────────────────────────
        turn += 1;
        let prompt_a = format_prompt(&config.system_prompt_a, summary_a.as_deref(), &history_a);
        let (resp_a, think_a, stats_a, debug_a) = run_and_stream(
            &binary_path, &model_path_str, &prompt_a, auto_ngl,
            color_a, header_color, stats_color, &config.agent_name_a,
            &model_params, config.show_stats, config.show_thinking, think_color,
        )?;
        let resp_a_clean = clean_response(&resp_a);
        history_a.push(Message { role: Role::Assistant, content: resp_a_clean.clone() });
        history_b.push(Message { role: Role::User,      content: resp_a_clean.clone() });

        let stats_str_a = if config.log_stats { stats_a.as_ref().map(format_stats) } else { None };
        let _ = append_to_log(
            &log_file_path, &config.agent_name_a, &resp_a_clean,
            if config.log_thinking && !think_a.is_empty() { Some(&think_a) } else { None },
            stats_str_a.as_deref(),
            if config.log_debug && !debug_a.is_empty() { Some(&debug_a) } else { None },
            &config.log_mode,
        );
        if history_a.len() > config.compress_threshold {
            compress_history(&binary_path, &model_path_str, auto_ngl,
                &mut summary_a, &mut history_a, &config.agent_name_a, &config.agent_name_b,
                &model_params, config.compress_threshold, config.compress_keep)?;
        }

        if limit.map(|l| turn >= l).unwrap_or(false) { break; }

        // ── Agent B ──────────────────────────────────────────────────────────
        turn += 1;
        let prompt_b = format_prompt(&config.system_prompt_b, summary_b.as_deref(), &history_b);
        let (resp_b, think_b, stats_b, debug_b) = run_and_stream(
            &binary_path, &model_path_str, &prompt_b, auto_ngl,
            color_b, header_color, stats_color, &config.agent_name_b,
            &model_params, config.show_stats, config.show_thinking, think_color,
        )?;
        let resp_b_clean = clean_response(&resp_b);
        history_a.push(Message { role: Role::User,      content: resp_b_clean.clone() });
        history_b.push(Message { role: Role::Assistant, content: resp_b_clean.clone() });

        let stats_str_b = if config.log_stats { stats_b.as_ref().map(format_stats) } else { None };
        let _ = append_to_log(
            &log_file_path, &config.agent_name_b, &resp_b_clean,
            if config.log_thinking && !think_b.is_empty() { Some(&think_b) } else { None },
            stats_str_b.as_deref(),
            if config.log_debug && !debug_b.is_empty() { Some(&debug_b) } else { None },
            &config.log_mode,
        );
        if history_b.len() > config.compress_threshold {
            compress_history(&binary_path, &model_path_str, auto_ngl,
                &mut summary_b, &mut history_b, &config.agent_name_b, &config.agent_name_a,
                &model_params, config.compress_threshold, config.compress_keep)?;
        }
    }

    let _ = execute!(stdout, SetForegroundColor(Color::DarkGrey), Print("\n--- Conversation finished. ---\n"), ResetColor);
    Ok(())
}
