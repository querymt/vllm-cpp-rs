use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::PathBuf;

use vllm_cpp::HuggingFaceModel;

#[derive(Debug)]
struct UsageError(String);

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for UsageError {}

#[derive(Debug, Eq, PartialEq)]
pub enum ModelSource {
    Local(PathBuf),
    Gguf {
        repo: String,
        filename: String,
        revision: Option<String>,
    },
    Safetensors {
        repo: String,
        revision: Option<String>,
    },
}

impl ModelSource {
    pub fn resolve(self) -> Result<PathBuf, Box<dyn Error>> {
        match self {
            Self::Local(path) => Ok(path),
            Self::Gguf {
                repo,
                filename,
                revision,
            } => {
                let resolver = HuggingFaceModel::gguf(repo, filename);
                let resolver = if let Some(revision) = revision {
                    resolver.revision(revision)
                } else {
                    resolver
                };
                Ok(resolver.progress(true).resolve()?)
            }
            Self::Safetensors { repo, revision } => {
                let resolver = HuggingFaceModel::safetensors(repo);
                let resolver = if let Some(revision) = revision {
                    resolver.revision(revision)
                } else {
                    resolver
                };
                Ok(resolver.progress(true).resolve()?)
            }
        }
    }
}

#[allow(dead_code)]
pub fn resolve_model(example: &str) -> Result<PathBuf, Box<dyn Error>> {
    parse_model_source(example, env::args_os().skip(1))?.resolve()
}

fn parse_model_source(
    example: &str,
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<ModelSource, UsageError> {
    let mut args = arguments.into_iter();
    let first = args.next().ok_or_else(|| usage(example))?;
    match first.to_str() {
        Some("local") => {
            let path = args.next().ok_or_else(|| usage(example))?;
            if path.is_empty() {
                return Err(usage(example));
            }
            require_end(example, &mut args)?;
            Ok(ModelSource::Local(path.into()))
        }
        Some("hf-gguf") => {
            let repo = required_utf8(example, &mut args)?;
            let filename = required_utf8(example, &mut args)?;
            let revision = optional_revision(example, &mut args)?;
            Ok(ModelSource::Gguf {
                repo,
                filename,
                revision,
            })
        }
        Some("hf-safetensors") => {
            let repo = required_utf8(example, &mut args)?;
            let revision = optional_revision(example, &mut args)?;
            Ok(ModelSource::Safetensors { repo, revision })
        }
        Some(value) if value.starts_with('-') => Err(usage(example)),
        _ => {
            if first.is_empty() {
                return Err(usage(example));
            }
            require_end(example, &mut args)?;
            Ok(ModelSource::Local(first.into()))
        }
    }
}

fn required_utf8(
    example: &str,
    args: &mut impl Iterator<Item = OsString>,
) -> Result<String, UsageError> {
    let value = args
        .next()
        .ok_or_else(|| usage(example))?
        .into_string()
        .map_err(|_| usage(example))?;
    if value.is_empty() {
        return Err(usage(example));
    }
    Ok(value)
}

fn optional_revision(
    example: &str,
    args: &mut impl Iterator<Item = OsString>,
) -> Result<Option<String>, UsageError> {
    let Some(flag) = args.next() else {
        return Ok(None);
    };
    if flag != OsStr::new("--revision") {
        return Err(usage(example));
    }
    let revision = required_utf8(example, args)?;
    require_end(example, args)?;
    Ok(Some(revision))
}

fn require_end(example: &str, args: &mut impl Iterator<Item = OsString>) -> Result<(), UsageError> {
    if args.next().is_some() {
        return Err(usage(example));
    }
    Ok(())
}

fn usage(example: &str) -> UsageError {
    UsageError(format!(
        "usage:\n  {example} <model-directory-or-gguf>\n  {example} local <model-directory-or-gguf>\n  {example} hf-gguf <repo> <filename> [--revision <revision>]\n  {example} hf-safetensors <repo> [--revision <revision>]"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn parses_local_aliases() {
        assert_eq!(
            parse_model_source("example", os(&["model.gguf"])).unwrap(),
            ModelSource::Local("model.gguf".into())
        );
        assert_eq!(
            parse_model_source("example", os(&["local", "model"])).unwrap(),
            ModelSource::Local("model".into())
        );
    }

    #[test]
    fn parses_hugging_face_sources_with_optional_revisions() {
        assert_eq!(
            parse_model_source("example", os(&["hf-gguf", "owner/model", "model.gguf"])).unwrap(),
            ModelSource::Gguf {
                repo: "owner/model".to_owned(),
                filename: "model.gguf".to_owned(),
                revision: None,
            }
        );
        assert_eq!(
            parse_model_source("example", os(&["hf-safetensors", "owner/model"])).unwrap(),
            ModelSource::Safetensors {
                repo: "owner/model".to_owned(),
                revision: None,
            }
        );
        assert_eq!(
            parse_model_source(
                "example",
                os(&[
                    "hf-gguf",
                    "owner/model",
                    "model.gguf",
                    "--revision",
                    "release",
                ]),
            )
            .unwrap(),
            ModelSource::Gguf {
                repo: "owner/model".to_owned(),
                filename: "model.gguf".to_owned(),
                revision: Some("release".to_owned()),
            }
        );
        assert_eq!(
            parse_model_source(
                "example",
                os(&["hf-safetensors", "owner/model", "--revision", "release",]),
            )
            .unwrap(),
            ModelSource::Safetensors {
                repo: "owner/model".to_owned(),
                revision: Some("release".to_owned()),
            }
        );
    }

    #[test]
    fn rejects_missing_extra_duplicate_and_misordered_arguments() {
        for args in [
            os(&[]),
            os(&[""]),
            os(&["local"]),
            os(&["local", ""]),
            os(&["model", "extra"]),
            os(&["hf-gguf", "owner/model"]),
            os(&["hf-gguf", "owner/model", ""]),
            os(&["hf-safetensors", ""]),
            os(&["hf-safetensors", "owner/model", "release"]),
            os(&["hf-safetensors", "owner/model", "--revision"]),
            os(&["hf-safetensors", "owner/model", "--revision", ""]),
            os(&[
                "hf-safetensors",
                "owner/model",
                "--revision",
                "one",
                "--revision",
                "two",
            ]),
            os(&[
                "hf-gguf",
                "--revision",
                "release",
                "owner/model",
                "model.gguf",
            ]),
        ] {
            assert!(parse_model_source("example", args).is_err());
        }
    }
}
